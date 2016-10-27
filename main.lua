require 'torch'
require 'xlua'
require 'paths'
require 'image'
local py = require 'fb.python'
local json = require 'json'

require 'transform'
optnet = require 'optnet'

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- Load optional libraries
xrequire('cunn')
xrequire('cudnn')

torch.setdefaulttensortype('torch.FloatTensor')
opts_mem = {inplace=true, reuseBuffers=true, mode='inference'}

local options = require 'options'
local data = require 'data'

local opts = options.parse(arg)

local activThresh = 0.002

data.checkIntegrity(opts)

-- Load the model
local model = nil
if opts.model == 'mpii' then
  model = torch.load('models/human_pose_mpii.t7')
else 
  model = torch.load('models/human_pose_lsp.t7')
end


if opts.useGPU then 
  if opts.usecudnn then
    cudnn.fastest = true
    cudnn.convert(model, cudnn)
  end
  model = model:cuda()
end

if opts.useGPU then
	optnet.optimizeMemory(model, torch.zeros(1,3,opts.res,opts.res):cuda(), opts_mem)
else
	optnet.optimizeMemory(model, torch.zeros(1,3,opts.res,opts.res), opts_mem)
end

model:evaluate()

raw_marking = io.open(opts.json):read("*a")
marking = json.decode(raw_marking)
n = 0
for _ in pairs(marking) do n = n + 1 end

-- Import python libraries and set pairs
py.exec([=[
import numpy as np
import cv2
import colorsys
import os.path as osp
import os
pairs = np.array([[1,2], [2,3], [3,7], [4,5], [4,7], [5,6], [7,9], [9,10], [14,9], [11,12], [12,13], [13,9], [14,15], [15,16]])-1
global image
global colors
N = 14
HSV_tuples = [(x*1.0/N, 1.0, 0.9) for x in range(N)]
colors = map(lambda x: [255.0*y for y in colorsys.hsv_to_rgb(*x)], HSV_tuples)
image = None
]=])

local predictions = torch.Tensor(n,16,2)

-- Set the progress bar
xlua.progress(0,n)

i = 1
for image_name, bboxes in pairs(marking) do
  if file_exists(opts.output..'/'..image_name)==false then

  local image_path = opts.imdir..image_name
  local img = image.load(image_path)

  py.exec([=[
global image
image=cv2.imread(im_path)]=], {im_path=image_path})

  for j=1,#bboxes do
    local preds = {}

    if bboxes[j]['score'] == nil or bboxes[j].score > opts.score_thresh then
      -- print(bboxes[j])
      
      local center = {bboxes[j].x + 0.5 * bboxes[j].w,
                      bboxes[j].y + 0.5 * bboxes[j].h}
      local scale = bboxes[j].h / 180
      local input = crop(img, center, scale, opts.res)

      input = (function() if opts.useGPU then return input:cuda() else return input end end)()
        
      -- Do the forward pass and get the predicitons
      local output = model:forward(input:view(1,3,opts.res,opts.res))
      
      output = applyFn(function (x) return x:clone() end, output)

      if opts.flip then
        local flippedOut = nil
        if opts.useGPU then
              flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res):cuda()))
        else
              flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res)))
        end
        flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
        output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut):float()
      else
        output = applyFn(function (x,y) return x:add(y):div(2) end, output, output):float()
      end
      output[output:lt(0)] = 0
      
      local preds_hm, preds_img = getPreds(output[1], center, scale)
      preds[#preds + 1] = preds_img

      py.exec([=[
global image
global colors
cv2.rectangle(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
for i in range(pairs.shape[0]):
  # plot only the visible joints
   if np.mean(output[pairs[i,0]]) > activThresh and np.mean(output[pairs[i,1]]) > activThresh:
    xx = preds[[pairs[i,0],pairs[i,1]],0]
    yy = preds[[pairs[i,0],pairs[i,1]],1]
    cv2.line(image, (xx[0], yy[0]), (xx[1], yy[1]), colors[i], thickness=2)
]=],{preds = preds_img:view(16,2), activThresh = activThresh, output = output:view(16,opts.res,opts.res),
pt1={bboxes[j].x, bboxes[j].y}, pt2={bboxes[j].x+bboxes[j].w-1, bboxes[j].y+bboxes[j].h-1}})

    end
  end
  py.exec([=[
global image

if not osp.exists(osp.dirname(path)):
  os.makedirs(osp.dirname(path))
cv2.imwrite(path, image)]=],{path=opts.output..'/'..(image_name:gsub("/", "_"))})
  collectgarbage()
  end
  xlua.progress(i, n)
  i = i + 1
end

-- if opts.eval then
--   distance = evaluate(predictions,valDataset)
--   calculateMetrics(distance,opts)
-- end
