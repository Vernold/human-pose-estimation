local C = {}

function C.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-useGPU', 1, 'Run the test on a GPU (0 is false)')
  cmd:option('-usecudnn', 1, 'Use cudnn (0 is false)')
  cmd:option('-model', 'mpii', 'Select the dataset to use: LSP/MPII')
  cmd:option('-res', 256, 'Input resolution')
  cmd:option('-json', '', 'Path to json file with detections')
  cmd:option('-imdir', '', 'Path to directory with images')
  cmd:option('-flip', 1, 'Average prediction between original and flipped image (0 is false)')
  cmd:option('-score_thresh', 0.9, 'Score threshold or filter detections')
  cmd:option('-output', 'output', 'Path to output directory')
  cmd:text()
  
  local opt = cmd:parse(arg or {})
  
  opt.model:lower()
  
  --assert(opt.model-='lsp' or opt.model=='mpii',"Only mpii and lsp are valid options")
  assert(opt.imdir~='' and opt.json~='', "You should specify images dir and json with marking")
  
  local function int2bool(value)
    if value < 1 then
      return false
    else
      return true
    end
  end

  opt.useGPU = int2bool(opt.useGPU)
  opt.usecudnn = int2bool(opt.usecudnn)
  opt.flip = int2bool(opt.flip)

  return opt
end

return C
