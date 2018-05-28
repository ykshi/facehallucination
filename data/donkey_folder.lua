--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = opt.data or os.getenv('DATA_ROOT') or '/data/local/imagenet-fetch/256'
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local loadSize   = opt.loadSize
local sampleSize = loadSize

local function loadImage(path)
   local input = image.load(path, opt.nc, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   -- local iW = input:size(3)
   -- local iH = input:size(2)
   -- if iW < iH then
   --    input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   -- else
   --    input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   -- end
   input = image.scale(input, loadSize[2], loadSize[1])
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local out
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2]
   local oH = sampleSize[1]
   if iH > oH and iw > oW then
     local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
     local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
     out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   else
     out = input
   end
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   -- if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.sampleHookTest = trainHook
   trainLoader.loadSize = {opt.nc, opt.loadSize[1], opt.loadSize[2]}
   trainLoader.sampleSize = {opt.nc, sampleSize[1], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      forceClasses = opt.forceTable,
      loadSize = {opt.nc, loadSize[1], loadSize[2]},
      sampleSize = {opt.nc, sampleSize[1], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.sampleHookTest = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
