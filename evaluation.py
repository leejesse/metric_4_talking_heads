from vis import calculate_ms_ssim, calculate_ssim, calculate_l1, calculate_psnr
from vis import averager
import pose_evaluation.extract as extract
import pose_evaluation.cmp_kp as cmp_kp
import pose_evaluation.cmp_kp as cmp
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Select an evaluation metric')
parser.add_argument('--metric', type=str, choices=['l1', 'ms-ssim', 'ssim', 'psnr', 'akd','aed'])
parser.add_argument('--gt_root', type=str, default='/home/yonggyu/pytorch-CartoonGAN/assets')
parser.add_argument('--recon_root', type=str, default='/home/yonggyu/pytorch-CartoonGAN/assets')
parser.add_argument('--gt_kp_path', type=str, default='pose_evaluation/pose_gt.pkl')
parser.add_argument('--recon_kp_path', type=str, default='pose_evaluation/pose_recon.pkl')
parser.add_argument('--gt_id_path', type=str, default='pose_evaluation/id_gt.pkl')
parser.add_argument('--recon_id_path', type=str, default='pose_evaluation/id_recon.pkl')
parser.add_argument('--image_shape', type=int, default=(256,256))

args = parser.parse_args()

if __name__ == "__main__":

    args.gt_root='/home/yonggyu/pytorch-CartoonGAN/assets'
    args.recon_root='/home/yonggyu/pytorch-CartoonGAN/assets'

    gt_path=glob.glob(os.path.join(args.gt_root,'**','*.jpg'), recursive=True)
    recon_path=glob.glob(os.path.join(args.recon_root,'**','*.jpg'), recursive=True)

    if args.metric == 'l1':
        averager(calculate_l1, gt_path, recon_path)

    elif args.metric == 'ms-ssim':
        averager(calculate_ms_ssim, gt_path, recon_path)
    
    elif args.metric == 'ssim':
        averager(calculate_ssim, gt_path, recon_path)
    
    elif args.metric == 'psnr':
        averager(calculate_psnr, gt_path, recon_path)
    
    elif args.metric == 'akd':
        extract.extract_face_pose(args.gt_root, False, args.image_shape,0).to_pickle(args.gt_kp_path)
        extract.extract_face_pose(args.recon_root, False, args.image_shape,0).to_pickle(args.recon_kp_path)
        cmp_kp.akd(args.gt_kp_path, args.recon_kp_path)
    
    elif args.metric == 'aed':
        #extract.extract_face_id(args.gt_root, False, args.image_shape,0).to_pickle(args.gt_id_path)
        #extract.extract_face_id(args.recon_root, False, args.image_shape,0).to_pickle(args.recon_id_path)
        #cmp.aed(args.gt_kp_path, args.recon_kp_path)
        pass

