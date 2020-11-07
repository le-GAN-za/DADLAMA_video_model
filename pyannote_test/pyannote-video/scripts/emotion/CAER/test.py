import argparse
import torch
import os
import json


import csv
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

EMOTION_TEMPLATE = ('{name:s},{emotion:s}/''\n')

def main(config):
    logger = config.get_logger('test')
    test = module_data.MyDataset.__getitem__
    # test = module_data


    # setup data_loader instances
    data_loader = config.init_obj('test_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config. resume, map_location=torch.device('cpu'))
    # checkpoint = torch.load(config. resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
   

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # original
    max_list = []
    name_list = []
    with torch.no_grad():
        for i, (data, target, path) in enumerate(tqdm(data_loader)):
            face, context, target, path = data['face'].to(device), data['context'].to(device), target.to(device), path
            output = model(face, context)
            
            #test line
            
            for num in range(len(output)):
                max_val = -100
                emotion_index = 0
                
                for k in range(7):
                    if max_val < output[num][k]:
                        max_val = output[num][k]
                        emotion_index = k
                mid_name = path[num].split('/')[-1]
                name_list.append(mid_name.split('.')[0])

                # name_list.append(path[num][13:])
                max_list.append(emotion_index)
                

            # computing loss, metrics on test set
            
            loss = loss_fn(output, target)
            batch_size = face.shape[0]
            total_loss += loss.item() * batch_size

            
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
               

    n_samples = len(data_loader.sampler)
    

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    
    emotion = {0:'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    count = 0
    emotion_count = {0: count, 1: count, 2: count, 3: count, 4: count, 5: count, 6: count}

    with open('./scripts/emotion/CAER/config.json','r') as con_r:
        json_data = json.load(con_r)
    
    path_data = json_data['test_loader']['args']['detect_file']
    emotion_file_name = path_data.split('/')[-1].split('.')[0]
    
    
    # text 파일 형식
    # with open('./data/emotion_detect/'+str(emotion_file_name)+'_emotion.txt','w') as fout:
    #     for i in range(len(max_list)):
    #         name = name_list[i]
    #         emotion_d = emotion[max_list[i]]
            
    #         emotion_count[max_list[i]] += 1

    #         fout.write(EMOTION_TEMPLATE.format(name = name, emotion = emotion_d))

    #     print(emotion_count)
        
    #     max_emotion = 0
    #     for i in range(len(emotion_count)):
    #         if max_emotion < emotion_count[i]:
    #             max_emotion = emotion_count[i]
    #             count = i
        
    #     print('이 장면의 전체적인 분위기:' + emotion[count])
    #     # fout.write('이 장면의 전체적인 분위기:' + emotion[count]+'\n')

    #     for i in range(7):
    #         fout.write(str(emotion[i])+':'+ str(emotion_count[i])+" ")
    
    with open('./output/emotion_detect/'+str(emotion_file_name)+'_emotion.csv','w', newline='') as fout:
        wr = csv.writer(fout)

        for i in range(len(max_list)):
            name = name_list[i]
            emotion_d = emotion[max_list[i]]
            
            emotion_count[max_list[i]] += 1

            wr.writerow([name, emotion_d])

        print(emotion_count)
        
        max_emotion = 0

        for i in range(len(emotion_count)):
            if max_emotion < emotion_count[i]:
                max_emotion = emotion_count[i]
                count = i
        
        print('이 장면의 전체적인 분위기:' + emotion[count])
        wr.writerow([emotion[count]])

        for i in range(7):
            wr.writerow([str(emotion[i]), str(emotion_count[i])])






if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', required=False, default='./CAER/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', required=False, default='./saved/models/CAERS_original_debug/1103_160110/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', required=False, default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
