
extra_loss=true
#extra_loss=false

exp_dir1=
exp_dir2=
#exp_dir3=

if [ "$extra_loss" = true ]
then
    #python plot_losses.py -e --dirs $exp_dir1 $exp_dir2 $exp_dir3
    python plot_losses.py -e --dirs $exp_dir1 $exp_dir2
    #python plot_losses.py -e --dirs $exp_dir1
else
    #python plot_losses.py --dirs $exp_dir1 $exp_dir2 $exp_dir3
    python plot_losses.py --dirs $exp_dir1 $exp_dir2
    #python plot_losses.py --dirs $exp_dir1
fi
