#!/bin/base
read -p "minute:" m
read -p "second:" s
for ((Time=m*60+s;Time>0;Time–))
do
	let m=Time/60
	let s=Time%60
	echo -n " m:m:s "
	echo -ne “\r”
	sleep 1
done
