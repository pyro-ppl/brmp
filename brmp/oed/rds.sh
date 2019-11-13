for i in {1..5}
do
    for m in {2..4}
    do
        python rds.py rand $m
        python rds.py oed $m
        python rds.py oed_alt $m
    done
done
