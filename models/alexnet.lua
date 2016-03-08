require 'loadcaffe'

function loadPremodel()
   if not paths.dirp('models/AlexNet') then
      print('=> Downloading AlexNet model weights')
     os.execute('mkdir models/AlexNet')
     local caffemodel_url = 'http://www.cs.bu.edu/groups/ivc/data/SOS/AlexNet_SalObjSub.caffemodel'
     local proto_url = 'https://gist.githubusercontent.com/jimmie33/0585ed9428dc5222981f/raw/ec5b38a662bfcd140dd1ce15a2949b38ef5630c2/deploy.prototxt'
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
   return pretrain
end

function createModel(nGPU)
   if not paths.dirp('models/AlexNet') then
      print('=> Downloading AlexNet model weights')
     os.execute('mkdir models/AlexNet')
     local caffemodel_url = 'http://www.cs.bu.edu/groups/ivc/data/SOS/AlexNet_SalObjSub.caffemodel'
     local proto_url = 'https://gist.githubusercontent.com/jimmie33/0585ed9428dc5222981f/raw/ec5b38a662bfcd140dd1ce15a2949b38ef5630c2/deploy.prototxt'
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

   if opt.trainType == 'transfer' then
      local model = nn.Sequential():add(classifier)
   elseif opt.trainType == 'finetune' then
      local model = nn.Sequential():add(pretrain):add(classifier)
   else
      print 'not supported type'
   end
   local model = nn.Sequential():add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return pretrain, model
end
