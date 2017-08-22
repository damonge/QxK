#!/bin/bash

echo "Getting data"
mkdir -p data
cd data
if [ ! -f mask.fits ] ; then
    wget irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/lensing/COM_CompMap_Lensing_2048_R2.00.tar
    tar -xvf COM_CompMap_Lensing_2048_R2.00.tar
    mv data/* .
    gunzip mask.fits.gz
    rm -r COM_CompMap_Lensing_2048_R2.00.tar data
fi
if [ ! -f DR12Q.fits ] ; then
    wget https://data.sdss.org/sas/dr13/env/BOSS_QSO/DR12Q/DR12Q.fits
fi
if [ ! -f DLA_DR12_v1.dat ] ; then
    echo "Please get the DLA catalog and put it in \"data\""
    cd ..
    exit 1
fi
cd ..

echo "Computing correlation functions"
nsims=1000
for nside in 2048 1024
do
    for nth in 16 27
    do
	prefix_dir=outputs_ns${nside}_nb${nth}
	if [ ! -f ${prefix_dir}/wth_qxk_all.npz ] ; then
	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py ${nth} ${nside} 0 ${nsims} 0
	fi
	if [ ! -f ${prefix_dir}_randp/wth_qxk_all.npz ] ; then
	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py ${nth} ${nside} 0 ${nsims} 1
	fi
	if [ ! -f ${prefix_dir}_wiener/wth_qxk_all.npz ] ; then
	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py ${nth} ${nside} 1 ${nsims} 0
	fi
    done
done

echo "Analyzing data"
for nside in 2048 1024
do
    for nth in 16 27
    do
	echo "Nside = ${nside}, n_theta = ${nth}"
#	python analysis_wth.py ${nth} ${nside} 0 ${nsims} 0 0
	echo "Nside = ${nside}, n_theta = ${nth}, randomized points"
#	python analysis_wth.py ${nth} ${nside} 0 ${nsims} 1 0
	echo "Nside = ${nside}, n_theta = ${nth}, Wiener-filtered"
#	python analysis_wth.py ${nth} ${nside} 1 ${nsims} 0 0
    done
done

python analysis_wth.py 16 2048 1 100 0 1
