import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils.comm import generate_model
from utils.metrics import Metrics


def test():
    print('loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)
    model = generate_model(opt)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'FPR', 'precision', 'F1', 'ZSI',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _FPR, _precision, _F1, _ZSI, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, FPR= _FPR, precision= _precision,
                            F1= _F1, ZSI= _ZSI, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )

    metrics_result = metrics.mean(total_batch)

    print("Test Result:")
    print('recall: %.3f, FPR: %.4f, precision: %.3f, F1: %.3f,'
              ' ZSI: %.3f, ACC_overall: %.3f, IoU_poly: %.3f, IoU_bg: %.3f, IoU_mean: %.3f'
              % (metrics_result['recall'], metrics_result['FPR'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['ZSI'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))


if __name__ == '__main__':

    if opt.mode == 'test':
        print('--- Cervix Test---')
        test()

    print('Done')
