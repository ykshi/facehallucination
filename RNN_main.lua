require 'dpnn'
require 'rnn'
require 'cutorch'
require 'nngraph'
require 'optim'
require 'image'

require 'VRMSEReward'
require 'SpatialGlimpse_inverse'
util = paths.dofile('util.lua')
-- nngraph.setDebug(true)
opt = lapp[[
   -b,--batchSize             (default 32)         batch size
   -r,--lr                    (default 0.0002)    learning rate

   --dataset                  (default 'folder')  imagenet / lsun / folder
   --nThreads                 (default 4)         # of data loading threads to use

   --beta1                    (default 0.5)       momentum term of adam
   --ntrain                   (default math.huge) #  of examples per epoch. math.huge for full dataset
   --display                  (default 0)         display samples while training. 0 = false
   --display_id               (default 10)        display window id.
   --gpu                      (default 1)         gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   --GAN_loss_after_epoch     (default 5)
   --name                     (default 'fullmodel')
   --checkpoints_name         (default '')        name of checkpoints for load
   --checkpoints_epoch        (default 0)         epoch of checkpoints for load
   --epoch                    (default 1)         save checkpoints every N epoch
   --nc                       (default 3)         number of input image channels (RGB/Grey)

   --niter                    (default 250)  maximum number of iterations

   --rewardScale              (default 1)     scale of positive reward (negative is 0)
   --rewardAreaScale          (default 4)     scale of aree reward
   --locatorStd               (default 0.11)  stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')

   --glimpseHiddenSize        (default 128)   size of glimpse hidden layer')
   --glimpsePatchSize         (default '60,45')     size of glimpse patch at highest res (height = width)')
   --glimpseScale             (default 1)     scale of successive patches w.r.t. original input image')
   --glimpseDepth             (default 1)     number of concatenated downscaled patches')
   --locatorHiddenSize        (default 128)   size of locator hidden layer')
   --imageHiddenSize          (default 512)   size of hidden layer combining glimpse and locator hiddens')
   --wholeImageHiddenSize     (default 256)   size of full image hidden size

   --pertrain_SR_loss         (default 2)     SR loss before training action
   --residual                 (default 1)     whether learn residual in each step
   --rho                      (default 25)    back-propagate through time (BPTT) for rho time-steps
   --hiddenSize               (default 512)   number of hidden units used in Simple RNN.
   --FastLSTM                 (default 1)     use LSTM instead of linear layer
   --BN                                       whether use BatchNormalization
   --save_im                                  whether save image on test
]]

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.ntrain == 'math.huge' then opt.ntrain = math.huge end
-- image size for dataloder, high-resolution and low-resolution
opt.loadSize = {128, 128}
opt.highResSize = {128, 128}
opt.lowResSize = {16, 16}
-- local patch size
local PatchSize = {}
PatchSize[1], PatchSize[2] = opt.glimpsePatchSize:match("([^,]+),([^,]+)")
opt.glimpsePatchSize = {}
opt.glimpsePatchSize[1] = tonumber(PatchSize[1])
opt.glimpsePatchSize[2] = tonumber(PatchSize[2])
opt.glimpseArea = opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2]
if opt.glimpseArea == opt.highResSize[1]*opt.highResSize[2] then
  opt.unitPixels = (opt.highResSize[2] - opt.glimpsePatchSize[2]) / 2
else
  opt.unitPixels = opt.highResSize[2] / 2
end
if opt.display == 0 then opt.display = false end -- lapp argparser cannot handel bool value 

opt.manualSeed = 123 --torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create train data loader
local DataLoader = paths.dofile('data/data.lua')
opt.data = '../lfw_funneled_dev_128/train/'
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
opt.data = '../lfw_funneled_dev_128/test/'
local dataTest = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') or name:find('Linear') then
      if m.weight then m.weight:normal(0.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    end
end

local nc = opt.nc
local rho = opt.rho
local lowResSize = opt.lowResSize
local highResSize = opt.highResSize

local SpatialBatchNormalization
if opt.BN then SpatialBatchNormalization = nn.SpatialBatchNormalization
else SpatialBatchNormalization = nn.Identity end
local SpatialConvolution = nn.SpatialConvolution

if opt.checkpoints_epoch and opt.checkpoints_epoch > 0 then
  nngraph.annotateNodes()
  print('Loading.. checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '_RNN.t7')
  model = torch.load('checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '_RNN.t7')
else
  ----------------------- locator net -----------------------
  -- Encode the (x,y) -- coordinate of last attended patch
  local locationSensor = nn.Sequential()
  locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
  locationSensor:add(nn.BatchNormalization(opt.locatorHiddenSize)):add(nn.ReLU(true))
  -- Encode the low-resolution input image
  local imageSensor = nn.Sequential()
  imageSensor:add(nn.View(-1):setNumInputDims(3))
  imageSensor:add(nn.Linear(nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
  imageSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  -- Encode the enhanced image in last step
  local imageErrSensor = nn.Sequential()
  imageErrSensor:add(nn.View(-1):setNumInputDims(3))
  imageErrSensor:add(nn.Linear(nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
  imageErrSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  -- rnn input
  glimpse = nn.Sequential()
  glimpse:add(nn.ParallelTable():add(locationSensor):add(imageErrSensor):add(imageSensor))
  glimpse:add(nn.JoinTable(1,1))
  glimpse:add(nn.Linear(opt.wholeImageHiddenSize+opt.locatorHiddenSize+opt.wholeImageHiddenSize, opt.imageHiddenSize))
  glimpse:add(nn.BatchNormalization(opt.imageHiddenSize)):add(nn.ReLU(true))
  glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))
  glimpse:add(nn.BatchNormalization(opt.hiddenSize)):add(nn.ReLU(true))
  -- rnn recurrent cell
  recurrent = nn.GRU(opt.hiddenSize, opt.hiddenSize)
  -- recurrent neural network
  local rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn.ReLU(true), 99999)
  -- output the coordinate of attended patch
  local locator = nn.Sequential()
  locator:add(nn.Linear(opt.hiddenSize, 2))
  locator:add(nn.Tanh()) -- bounds mean between -1 and 1
  locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
  locator:add(nn.HardTanh()) -- bounds sample between -1 and 1, while reinforce recieve no gradInput
  locator:add(nn.MulConstant(opt.unitPixels*2/highResSize[2]))
  
  ----------------------- SR net -----------------------
  -- globally encode the attended patch
  local SR_patch_fc = nn.Sequential()
  SR_patch_fc:add(nn.JoinTable(1,3))
  SR_patch_fc:add(nn.View(-1):setNumInputDims(3))
  SR_patch_fc:add(nn.Linear(nc*2*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2],256)):add(nn.ReLU(true))
  SR_patch_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  SR_patch_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_patch_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- globally encode the image
  local SR_img_fc = nn.Sequential()
  SR_img_fc:add(nn.JoinTable(1,3))
  SR_img_fc:add(nn.View(-1):setNumInputDims(3))
  SR_img_fc:add(nn.Linear(nc*2*highResSize[1]*highResSize[2],256)):add(nn.ReLU(true))
  SR_img_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  SR_img_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_img_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- transform the hidden of RNN
  local SR_fc = nn.Sequential()
  SR_fc:add(nn.Linear(opt.hiddenSize,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- fully-convolution network for SR
  local SRnet = nn.Sequential()
  SRnet:add(nn.JoinTable(1,3))
  SRnet:add(SpatialConvolution(nc*5, 16, 5, 5, 1, 1, 2, 2))
  SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(16, 32, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(32, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 32, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(32, 16, 5, 5, 1, 1, 2, 2))
  SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(16, nc, 5, 5, 1, 1, 2, 2))

  -- nngraph build model
  -- input: {loc_prev, image_pre, image}
  -- output: {loc, image_next}
  local loc_prev = nn.Identity()()
  local image_pre = nn.Identity()()
  local image = nn.Identity()()
  local visited_map_pre = nn.Identity()() -- used for record the attened area
  local onesTensor = nn.Identity()()

  local h = rnn({loc_prev,image_pre,image})
  local loc = locator(h)
  local visited_map = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)({visited_map_pre, onesTensor, loc})
  local patch = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image, loc})
  local patch_pre = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image_pre, loc})
  local SR_patch_fc_o = SR_patch_fc({patch, patch_pre})
  local SR_img_fc_o = SR_img_fc({image, image_pre})
  local SR_fc_o = SR_fc(h)
  local hr_patch = SRnet({patch, patch_pre, SR_patch_fc_o, SR_img_fc_o, SR_fc_o})
  if opt.residual then hr_patch = nn.Tanh()(nn.CAddTable()({hr_patch,patch_pre})) end
  local image_next = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({image_pre,hr_patch,loc})
  
  nngraph.annotateNodes()
  model = nn.gModule({loc_prev,image_pre,visited_map_pre,onesTensor,image}, {loc, image_next, visited_map})
  model:apply(weights_init)
  model.name = 'fullmodel'
  model = nn.Recursor(model, opt.rho)
end
gt_glimpse = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)
baseline_R = nn.Sequential()
baseline_R:add(nn.Add(1))
local REINFORCE_Criterion = nn.VRMSEReward(model, opt.rewardScale, opt.rewardAreaScale)
local MSEcriterion = nn.MSECriterion()
---------------------------------------------------------------------------
optimState = {
learningRate = opt.lr,
beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local outputs
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   model:cuda()
   baseline_R:cuda()
   MSEcriterion:cuda();      REINFORCE_Criterion:cuda();
   gt_glimpse:cuda()
end
model:forget()
local parameters, gradParameters = model:getParameters()
thin_model = model:sharedClone() -- used for save checkpoint
local a, b = thin_model:getParameters()
print(parameters:nElement())
print(gradParameters:nElement())

testLogger = optim.Logger(paths.concat(opt.name, 'test.log'))
testLogger:setNames{'MSE (training set)', 'PSNR (test set)'}
testLogger.showPlot = false

if opt.display then disp = require 'display' end

local fx = function(x)
  gradParameters:zero()
  model:forget()

  --fetch data
  highRes, idLabel = data:getBatch()
  lowRes = highRes:clone()
  for imI = 1, highRes:size(1) do
    temp = image.scale(highRes[imI], lowResSize[2], lowResSize[1])
    lowRes[imI] = image.scale(temp, highResSize[2], highResSize[1], 'bicubic')
  end
  highRes = highRes:cuda()
  lowRes = lowRes:cuda()
  idLabel = idLabel:cuda()

  local zero_loc = torch.zeros(opt.batchSize,2)
  local zero_dummy = torch.zeros(opt.batchSize,1)
  local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
  local visited_map0 = torch.zeros(opt.batchSize,1,highResSize[1],highResSize[2])
  zero_loc = zero_loc:cuda()
  zero_dummy = zero_dummy:cuda()
  ones = ones:cuda()
  visited_map0 = visited_map0:cuda()

  local dl = {}
  local inputs = {}
  outputs = {}
  gt = {}
  err_l = 0
  err_g = 0
  
  -- input: {loc_prev, image_pre, visited_map_prev, ones, image}
  -- output: {loc, image_next, visited_map_next}
  for t = 1,rho do
    if t == 1 then inputs[t] = {zero_loc, lowRes, visited_map0, ones, lowRes}
    else
      inputs[t] = outputs[t-1]
      table.insert(inputs[t], ones)
      table.insert(inputs[t], lowRes)
    end

    outputs[t] = model:forward(inputs[t])
    gt[t] = gt_glimpse:forward{highRes, outputs[t][1]}:clone()

    -- local MSE loss
    err_l = err_l + MSEcriterion:forward(outputs[t][2], highRes)
    dl[t] = MSEcriterion:backward(outputs[t][2], highRes):clone()
  end

  local curbaseline_R = baseline_R:forward(zero_dummy)
  err_g = REINFORCE_Criterion:forward({outputs[rho][2], outputs[rho][3], curbaseline_R}, highRes)
  
  --backward sequence
  local dg = REINFORCE_Criterion:backward({outputs[rho][2], outputs[rho][3], curbaseline_R}, highRes)
  
  for t = rho,1,-1 do
    -- zero_loc & visited_map0 are zero tensor, which is ok used as gradOutput in this case
    model:backward(inputs[t], {zero_loc, dl[t], visited_map0})
  end

  -- update baseline reward
  baseline_R:zeroGradParameters()
  baseline_R:backward(zero_dummy, dg[3])
  baseline_R:updateParameters(0.01)
  return err_g, gradParameters
end

function test()
  psnr = 0
  model:evaluate()

  paths.mkdir(opt.name)
  for st = 1,dataTest:size(),opt.batchSize do
    model:forget()
    xlua.progress(st,dataTest:size())
    --fetch data
    local i2, quantity
    if st + opt.batchSize > dataTest:size() then i2 = dataTest:size() 
    else i2 = st + opt.batchSize - 1 end
    quantity = i2 - st + 1
    highRes, impath = dataTest:getIndice({st,i2})
    lowRes = highRes:clone()
    for imI = 1, highRes:size(1) do
      temp = image.scale(highRes[imI], lowResSize[2], lowResSize[1])
      lowRes[imI] = image.scale(temp, highResSize[2], highResSize[1], 'bicubic')
    end
    highRes = highRes:cuda()
    lowRes = lowRes:cuda()

    local zero_loc = torch.zeros(highRes:size(1),2):cuda()
    local ones = torch.ones(highRes:size(1),1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):cuda()
    local visited_map0 = torch.zeros(highRes:size(1),1,highResSize[1],highResSize[2]):cuda()
    local output_t
    local input_t
    for t = 1,rho do
      if t == 1 then input_t = {zero_loc, lowRes, visited_map0, ones, lowRes}
      else
        input_t = {}
        for i = 1, #output_t do input_t[i] = output_t[i]:clone() end
        table.insert(input_t, ones)
        table.insert(input_t, lowRes)
      end
      output_t = model:forward(input_t)
    end

    for i = 1,quantity do
      -- 10* log10( 255^2 / (mse * (255/2)^2) )
      psnr = psnr + 10 * math.log10(4 / MSEcriterion:forward(output_t[2][i], highRes[i]))
      if opt.save_im then
        local img = output_t[2][i]
        img:add(1):div(2)
        image.save(opt.name..'/'..paths.basename(impath[i]), img)
      end
    end
  end
  psnr = psnr / dataTest:size()
  print(psnr)
  model:training()

  if testLogger then
    paths.mkdir(opt.name)
    testLogger:add{err_g, psnr}
    testLogger:style{'-','-'}
    testLogger:plot()
  end
end

-- train
epoch = opt.checkpoints_epoch and opt.checkpoints_epoch or 0
while epoch < opt.niter do
   epoch = epoch+1
   epoch_tm:reset()
   test()
   local counter = 0
   local counter_test = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      collectgarbage()
      tm:reset()

      -- update model
      optim.adam(fx, parameters, optimState)
      a:copy(parameters)
      
      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
        local loc_im = torch.Tensor(opt.batchSize,nc,highResSize[1],highResSize[2])
        local p = torch.Tensor(opt.rho,nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
        for i = 1,opt.batchSize do
          loc_im[i] = outputs[rho][2][i]:clone():float()
        end
        for t = 1,#gt do
          p[t] = gt[t][1]:clone():float()
        end
        disp.image(loc_im, {win=opt.display_id, title=opt.name..'_output'})
        disp.image(lowRes, {win=opt.display_id+1, title=opt.name..'_input'})
        disp.image(highRes, {win=opt.display_id+2, title=opt.name..'_gt'})
        disp.image(p, {win=opt.display_id+3, title=opt.name..'_gtPatch'})
        disp.image(outputs[rho][3], {win=opt.display_id+4, title=opt.name..'_VisitedMap'})
     end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f '
            .. '  Global_MSE: %.4f Local_MSE: %.4f PSNR: %.4f'):format(
            epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            err_g or -1, err_l and err_l / rho or -1, psnr and psnr or -1))
      end
    end
    paths.mkdir('checkpoints')

    if epoch % opt.epoch == 0 then
      torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_RNN.t7', thin_model)
    end

    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
      epoch, opt.niter, epoch_tm:time().real))
end

