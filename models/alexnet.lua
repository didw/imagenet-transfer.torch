require 'loadcaffe'
function createModel(nGPU)
   if not paths.dirp('models/AlexNet') then
      print('=> Downloading AlexNet model weights')
     os.execute('mkdir models/AlexNet')
     local caffemodel_url = ''
     local proto_url = ''
     os.execute('wget --output-document models/AlexNet/AlexNet_SalObjSub.caffemodel ' .. caffemodel_url)
     os.execute('wget --output-document models/AlexNet/deploy.prototxt '              .. proto_url)
   end
   
   local proto = 'models/AlexNet/deploy.prototxt'
   local caffemodel = 'models/AlexNet/AlexNet_SalObjSub.caffemodel'
   
   if opt.backend == 'cudnn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cudnn')   
   elseif opt.backend == 'cnn2' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cnn2')
   else
      print 'AlexNet only support cudnn and cnn2'
      exit(1)   
   end
   
   for i=24,20,-1 do
      pretrain:remove(i)
   end
   
   local classifier = nn.Sequential()
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   local model = nn.Sequential():add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return pretrain, model
end
