for i in {1..20}
do
    python rds.py -m 2 --num-samples 6000 --num-epochs 200 -lr 0.0001 --weight-decay 0.5 --opt-method SGD --interval-method adapt
done
