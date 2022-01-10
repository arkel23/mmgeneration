#! /bin/bash
base="tools/dataset/results"
declare fnArray=("10037 1049001 face_1024_208_28")
#declare fnArray=("madoka test")
declare preprocessModesArray=("linear" "aoda" "xdog_serial_0.5")

base2="work_dirs/experiments/pix2pix_moe"
declare -a ckptsArray=("${base2}_all/best_fid_iter_160000.pth" "${base2}_aoda/best_fid_iter_190000.pth"
"${base2}_canny/best_fid_iter_80000.pth" "${base2}_dl/best_fid_iter_190000.pth"
"${base2}_linear/best_fid_iter_10000.pth" "${base2}_pidinet_0/best_fid_iter_100000.pth"
"${base2}_pidinet_-1/best_fid_iter_110000.pth" "${base2}_pidinet_1/best_fid_iter_140000.pth"
"${base2}_pidinet_2/best_fid_iter_170000.pth" "${base2}_pidinet_3/best_fid_iter_160000.pth"
"${base2}_sketchkeras/best_fid_iter_110000.pth" "${base2}_trad/best_fid_iter_190000.pth"
"${base2}_xdog/best_fid_iter_10000.pth"  "${base2}_xdog_serial0.3/best_fid_iter_170000.pth"
"${base2}_xdog_serial0.4/best_fid_iter_10000.pth" "${base2}_xdog_serial0.5/best_fid_iter_10000.pth"
"${base2}_xdog_th/best_fid_iter_190000.pth")

#declare -a ckptsArray=("${base2}_all/best_fid_iter_160000.pth" "${base2}_dl/best_fid_iter_190000.pth" "${base2}_trad/best_fid_iter_190000.pth" "${base2}_xdog_serial0.3/best_fid_iter_170000.pth")

cfg="configs/custom/moe/pix2pix/pix2pix_moe_linear.py"

for ckpt in ${ckptsArray[@]}; do
    for fn in ${fnArray[@]}; do
        for mode in ${preprocessModesArray[@]}; do
            img="${base}/${fn}_${mode}.jpg"
            echo "python $cfg $ckpt $img"
            python demo/translation_demo_v2.py $cfg $ckpt $img
        done
    done
done
