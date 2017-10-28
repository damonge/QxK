#!/bin/bash

echo "Getting data"
mkdir -p data
cd data

if [ ! -f mask.fits ] ; then
    echo " Downloading CMB lensing maps"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/lensing/COM_CompMap_Lensing_2048_R2.00.tar
    tar -xvf COM_CompMap_Lensing_2048_R2.00.tar
    mv data/* .
    gunzip mask.fits.gz
    rm -r COM_CompMap_Lensing_2048_R2.00.tar data
fi

if [ ! -f DR12Q.fits ] ; then
    echo " Downloading quasar catalog"
    wget https://data.sdss.org/sas/dr13/env/BOSS_QSO/DR12Q/DR12Q.fits
fi

if [ ! -f DLA_DR12_v2.dat ] ; then
    echo " Downloading DLA N12 data"
    wget www2.iap.fr/users/noterdae/DLA/DLA_DR12_v2.tgz
    tar -xvf DLA_DR12_v2.tgz
    rm LOS_DR12_v2.dat DLA_DR12_v2.tgz
fi

if [ ! -f table3.dat ] ; then
    echo " Downloading DLA N12 data"
    wget https://www.dropbox.com/sh/p4la9s8rdsj64xm/AAC2A5inG__5hwRQjO4JkjeXa/ascii_catalog/table3.dat?dl=0
    mv table3.dat\?dl\=0 table3.dat
fi

cd ..

echo "Reformatting data"
python reformat.py

#nsims=1000
nsims=1000
thmax=3.0
echo "Computing 2-point functions"
for nside in 2048
do
    #Correlation functions
    for thm in 3.0
    do
	for nth in 16 32 48
	do
	    prefix_dir=outputs_thm${thm}_ns${nside}_nb${nth}
	    if [ ! -f ${prefix_dir}/wth_qxk_all.npz ] ; then
		echo addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py ${thm} ${nth} ${nside} 0 ${nsims}
	    fi
	    if [ ! -f ${prefix_dir}_wiener/wth_qxk_all.npz ] ; then
		echo addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py ${thm} ${nth} ${nside} 1 ${nsims}
	    fi
	done
    done

    #Power spectra
    for nlb in 25 40 50 75
    do
	for aposcale in 0.0 0.1
	do
	    prefix_dir=outputs_ell2_2002_ns${nside}_nlb${nlb}_apo${aposcale}00
	    if [ ! -f ${prefix_dir}/cl_qxk_all.npz ] ; then
		addqueue -q cmb -s -n 1x12 -m 1.8 /usr/local/shared/python/2.7.6-gcc/bin/python run_cls.py 2 2002 ${nlb} ${nside} ${nsims} ${aposcale}
	    fi
	done
    done
done

echo "Analysing data"
python analysis_cls.py 40 2048 1000 0.0 1000 1

<<COMMENT

echo "Computing correlation functions"
nsims=1000

#for nside in 2048 1024
for nside in 2048
do
    for nth in 16 32 48
    do
#	prefix_dir=outputs_thm10.0_ns${nside}_nb${nth}
	prefix_dir=outputs_thm3.0_ns${nside}_nb${nth}_hisn
	if [ ! -f ${prefix_dir}/wth_qxk_all.npz ] ; then
	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py 3.0 ${nth} ${nside} 0 ${nsims} 0 1
	fi
#	if [ ! -f ${prefix_dir}_randp/wth_qxk_all.npz ] ; then
#	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py 3.0 ${nth} ${nside} 0 ${nsims} 1 1
#	fi
	if [ ! -f ${prefix_dir}_wiener/wth_qxk_all.npz ] ; then
	    addqueue -q cmb -s -n 1x12 -m 1 /usr/local/shared/python/2.7.6-gcc/bin/python run_correlations.py 3.0 ${nth} ${nside} 1 ${nsims} 0 1
	fi
    done
done

echo "Analyzing data"
thmax=1.0
for nside in 2048 1024
do
    for nth in 16 27
    do
	echo "Nside = ${nside}, n_theta = ${nth}, thmax=${thmax}"
	python analysis_wth.py ${thmax} ${nth} ${nside} 0 ${nsims} 0 0
#	echo "Nside = ${nside}, n_theta = ${nth}, randomized points"
#	python analysis_wth.py 1.0 ${nth} ${nside} 0 ${nsims} 1 0
	echo "Nside = ${nside}, n_theta = ${nth}, thmax=${thmax}, Wiener-filtered"
	python analysis_wth.py ${thmax} ${nth} ${nside} 1 ${nsims} 0 0
    done
done

thmax=3.0
for nside in 2048 1024
do
    for nth in 16 48
    do
	echo "Nside = ${nside}, n_theta = ${nth}, thmax=${thmax}"
	python analysis_wth.py ${thmax} ${nth} ${nside} 0 ${nsims} 0 0
#	echo "Nside = ${nside}, n_theta = ${nth}, randomized points"
#	python analysis_wth.py 1.0 ${nth} ${nside} 0 ${nsims} 1 0
	echo "Nside = ${nside}, n_theta = ${nth}, thmax=${thmax}, Wiener-filtered"
	echo "python analysis_wth.py ${thmax} ${nth} ${nside} 1 ${nsims} 0 0"
	python analysis_wth.py ${thmax} ${nth} ${nside} 1 ${nsims} 0 0
    done
done

#python analysis_wth.py 16 1024 1 100 0 1
#python analysis_wth.py 16 1024 1 100 0 0
COMMENT
