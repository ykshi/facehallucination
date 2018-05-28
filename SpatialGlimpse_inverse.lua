------------------------------------------------------------------------
--[[ SpatialGlimpse ]]--
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- a glimpse is the concatenation of down-scaled cropped images of
-- increasing scale around a given location in a given image.
-- input is a pair of Tensors: {image, location}
-- locations are x,y coordinates of the center of cropped patches.
-- Coordinates are between -1,-1 (top-left) and 1,1 (bottom right)
-- output is a batch of glimpses taken in image at location (x,y)
-- glimpse size is {height, width}, or width only if square-shaped
-- depth is number of patches to crop per glimpse (one patch per scale)
-- Each successive patch is scale x size of the previous patch
------------------------------------------------------------------------
local SpatialGlimpse, parent = torch.class("nn.SpatialGlimpse_inverse", "nn.Module")

function SpatialGlimpse:__init(size, residual)
   require 'nnx'
   if torch.type(size)=='table' then
      self.height = size[1]
      self.width = size[2]
   else
      self.width = size
      self.height = size
   end
   self.residual = residual

   assert(torch.type(self.width) == 'number')
   assert(torch.type(self.height) == 'number')

   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function SpatialGlimpse:updateOutput(inputTable)
  assert(torch.type(inputTable) == 'table')
  assert(#inputTable == 3)
  -- inputTable_ index the table includes global and local image
  local input_g, input_l, location = unpack(inputTable)
  local glimpseWidth = self.width
  local glimpseHeight = self.height
  self._pad = self._pad or input_g.new()

  --input_g, input_l, location = self:toBatch(input_g, 3), self:toBatch(input_l, 3), self:toBatch(location, 1)
  assert(input_g:dim() == 4 and input_l:dim() == 4 and location:dim() == 2)
  local h, w = input_g:size(3), input_g:size(4)
  self.output:resizeAs(input_g)
  for sampleIdx = 1,self.output:size(1) do
    local dst = self.output[sampleIdx]
    local inputSample_g = input_g[sampleIdx]
    local inputSample_l = input_l[sampleIdx]
    local yx = location[sampleIdx]
    -- (-1,-1) top left corner, (1,1) bottom right corner of image
    local y, x = yx:select(1,1), yx:select(1,2)
    -- rescale to (0,0) ~ (1,1)
    y, x = (y+1)/2, (x+1)/2

    -- add zero padding (glimpse could be partially out of bounds)
    local padWidth = math.floor((glimpseWidth-1)/2)
    local padHeight = math.floor((glimpseHeight-1)/2)
    self._pad:resize(input_g:size(2), input_g:size(3)+padHeight*2, input_g:size(4)+padWidth*2):zero()
    local center = self._pad:narrow(2,padHeight+1,input_g:size(3)):narrow(3,padWidth+1,input_g:size(4))
        
    local h, w = self._pad:size(2)-glimpseHeight, self._pad:size(3)-glimpseWidth
    local y, x = math.min(h,math.max(0,y*h)),  math.min(w,math.max(0,x*w))
    local patch = self._pad:narrow(2,y+1,glimpseHeight):narrow(3,x+1,glimpseWidth)

    if not self.residual then 
      center:copy(inputSample_g)
    end
    patch:copy(inputSample_l)

    dst:copy(center)
  end
  
  self.output = self.output
  return self.output
end

function SpatialGlimpse:updateGradInput(inputTable, gradOutput)
  local input_g, input_l, location = unpack(inputTable)
  local gradInput_g, gradInput_l, gradLocation = unpack(self.gradInput)
  -- input_g, input_l, location = self:toBatch(input_g, 3), self:toBatch(input_l, 3), self:toBatch(location, 1)
  -- gradOutput = self:toBatch(gradOutput, 3)

  gradInput_g:resizeAs(input_g):zero()
  gradInput_l:resizeAs(input_l):zero()
  gradLocation:resizeAs(location):zero() --no backprop through location

  local glimpseWidth = self.width
  local glimpseHeight = self.height
  for sampleIdx=1,gradOutput:size(1) do
    local gradOutputSample = gradOutput[sampleIdx]
    local gradInputSample_g = gradInput_g[sampleIdx]
    local gradInputSample_l = gradInput_l[sampleIdx]
    local yx = location[sampleIdx] --height, width

    --(-1,-1) index top left corner, (1,1) bottom right corner of image
    local y, x = yx:select(1,1), yx:select(1,2)

    --(0,0) ,(1,1)
    y, x = (y+1)/2, (x+1)/2
    
    -- add zero padding (glimpse could be partially out of bounds)
    local padWidth = math.floor((glimpseWidth-1)/2)
    local padHeight = math.floor((glimpseHeight-1)/2)
    self._pad:resize(input_g:size(2), input_g:size(3)+padHeight*2, input_g:size(4)+padWidth*2):zero()
    local center = self._pad:narrow(2,padHeight+1,input_g:size(3)):narrow(3,padWidth+1,input_g:size(4))
    center:copy(gradOutputSample)

    -- crop it
    local h, w = self._pad:size(2)-glimpseHeight, self._pad:size(3)-glimpseWidth
    local y, x = math.min(h,math.max(0,y*h)),  math.min(w,math.max(0,x*w))

    if not self.residual then
      gradInputSample_g:add(gradOutputSample)
    end
    gradInputSample_l:add(self._pad:narrow(2,y+1,glimpseHeight):narrow(3,x+1,glimpseWidth))
  end

  -- self.gradInput[1] = self.fromBatch(gradInput_g, 1)
  -- self.gradInput[2] = self.fromBatch(gradInput_l, 1)
  -- self.gradInput[3] = self.fromBatch(gradLocation, 1)
  self.gradInput = {gradInput_g, gradInput_l, gradLocation}

  return self.gradInput
end
