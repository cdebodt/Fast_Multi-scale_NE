#! python3
# -*-coding:Utf-8 -*

import sklearn.datasets
import sklearn.manifold
import time

import fmsnepy

##############################
# Demo presenting how to use the main functions of this file.
####################

if __name__ == '__main__':
    print("==============================================")
    print("===== Starting the demo of fast_ms_ne.py =====")
    print("==============================================")

    # List of tuples. There is one tuple per considered data set in this demo. The first element of each tuple is a function enabling to load the data set, while the second element of each tuple is a string storing a name for the associated data set.
    L_data = [(sklearn.datasets.load_digits, 'Digits'), (lambda: sklearn.datasets.make_blobs(n_samples=11000, n_features=12, centers=22, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=3), 'Blobs')]
    n_data = len(L_data)

    # Boolean. Whether or not to plot the LD embeddings as they are computed in the demo.
    plot_emb = True

    # Largest neighborhood size to consider when employing the 'red_rnx_auc' function for the reduced quality assessment in the demo.
    Kup = 10000

    # For each data set
    for idx_data, data_t in enumerate(L_data):
        ###
        ###
        ###
        # Load function and name of the current data set
        data_load, data_name = data_t
        print("=====")
        print("==== Data set #{i}/{n_data}: {data_name}".format(i=idx_data+1, n_data=n_data, data_name=data_name))
        print("=====")

        ###
        ###
        ###
        print('- Loading the HD data set')
        # TIP: to change the employed data set, you just need to
        # modify the next code line to provide different values for
        # X_hds and labels. Afterwards, only X_hds is employed to
        # compute the LD embeddings. The labels are only used to plot
        # the obtained LD embeddings using colors.
        D_data = data_load()
        if isinstance(D_data, dict):
            X_hds, labels = D_data['data'], D_data['target']
        elif isinstance(D_data, tuple):
            X_hds, labels = D_data
        else:
            raise ValueError("Error in the demo of module {module_name}: the data set '{data_name}' #{i}/{n_data} cannot be loaded.".format(module_name=module_name, data_name=data_name, i=idx_data+1, n_data=n_data))
        # Number of samples and dimension of the HD data set
        N_samp, M_HD = X_hds.shape
        print("Number of data samples: {N_samp}".format(N_samp=N_samp))
        print("HDS dimension: {M_HD}".format(M_HD=M_HD))
        # Targeted dimension of the LD embeddings
        dim_LDS = 2
        print("Targeted LDS dimension: {dim_LDS}".format(dim_LDS=dim_LDS))
        # Whether the currently considered data set is big in terms of
        # its number of samples or not.
        big_data = (N_samp >= 10000)
        if big_data:
            print('The data set is big in terms of its number of samples.')
            print('Multi-scale SNE, multi-scale t-SNE and t-SNE are hence not applied; only their fast versions are employed (fast multi-scale SNE, fast multi-scale t-SNE and Barnes-Hut t-SNE).')
            print('The reduced DR quality is evaluated; it means that the R_{NX}(K) curve is computed only for K=1 to Kup={Kup}, and that the AUC refers to the area under this reduced curve, with a log scale for K, instead of the full one for K=1 to N-2={v}, with N being the number of data samples.'.format(Kup=Kup, v=N_samp-2, NX='{NX}'))
        else:
            print('The data set is moderate in terms of its number of samples.')
            print('Multi-scale SNE, multi-scale t-SNE and t-SNE are hence applied, as well as their fast versions (fast multi-scale SNE, fast multi-scale t-SNE and Barnes-Hut t-SNE).')
            print('The DR quality is completely evaluated; the R_{NX}(K) curve is computed for K=1 to N-2={v}, with N being the number of data samples, and the AUC refers to the area under this curve with a log scale for K.'.format(v=N_samp-2, NX='{NX}'))
        print('===')
        print('===')
        print('===')

        ###
        ###
        ###
        # fmsnepy.eucl_dist_matr() is used to compute a 2-D numpy
        # array containing the pairwise distances in a data set,
        # if it is not too big in terms of its number of
        # samples. This function is used to compute the HD and LD
        # distances for the DR quality assessment when the data
        # set is of moderate size.
        #
        # Note that in all DR methods employed in this code
        # (multi-scale SNE, multi-scale t-SNE, t-SNE, fast
        # multi-scale SNE, fast multi-scale t-SNE, Barnes-Hut
        # t-SNE), the LD embedding is computed using Euclidean
        # distances in the LD space.


        # Lists to provide as parameters to viz_qa, to visualize the
        # DR quality assessment as conducted in [1].
        L_rnx, Lmarkers, Lcols, Lleg_rnx, Lls, Lmedw, Lsdots = [], [], [], [], [], [], []

        ###
        ###
        ###
        # If the data set is not too big, we can compute all the
        # pairwise HD distances between its samples.
        if not big_data:
            print('- Computing the pairwise Euclidean distances in the HD data set')
            t0 = time.time()
            dm_hd = fmsnepy.eucl_dist_matr(X_hds)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('===')
            print('===')
            print('===')

        ###
        ###
        ###
        # Initialization type of the LD embedding. Check the
        # 'init_ld_emb' function for details. Note that you can
        # provide the LD coordinates to use for the initialization by
        # setting init_ld_emb to a 2-D numpy.ndarray containing the
        # initial LD positions, with one example per row and one LD
        # dimension per column, init_ld_emb[i,:] containing the
        # initial LD coordinates related to the HD sample X_hds[i,:].
        init_ld_emb = 'pca'

        ###
        ###
        ###
        # Applying multi-scale t-SNE if the data set is not too big, i.e. it is limited to a few thousands samples.
        if not big_data:
            print('- Applying multi-scale t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
            if data_name == 'Digits':
                print('This takes a few seconds (i.e., around 17 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            X_ld_mstsne = fmsnepy.mstsne(X_hds=X_hds, n_components=dim_LDS, init=init_ld_emb, rand_state=fmsnepy.np.random.RandomState(2))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

            ###
            ###
            ###
            print('- Evaluating the DR quality of the LD embedding obtained using multi-scale t-SNE')
            if data_name == 'Digits':
                print('This takes a few seconds (i.e., around 1 second with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            rnx_mstsne, auc_mstsne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_mstsne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_mstsne, 4)))

            # Updating the lists for viz_qa
            L_rnx.append(rnx_mstsne)
            Lmarkers.append('^')
            Lcols.append('blue')
            Lleg_rnx.append('Ms $t$-SNE')
            Lls.append('solid')
            Lmedw.append(0.5)
            Lsdots.append(10)

            ###
            ###
            ###
            if plot_emb:
                print('- Plotting the LD embedding obtained using multi-scale t-SNE')
                print('If a figure is shown, close it to continue.')
                # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
                fmsnepy.viz_2d_emb(X=X_ld_mstsne, vcol=labels, tit='LD embedding Ms $t$-SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
                print('===')
                print('===')
                print('===')

        ###
        ###
        ###
        # Applying multi-scale SNE if the data set is not too big, i.e. it is limited to a few thousands samples.
        if not big_data:
            print('- Applying multi-scale SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
            if data_name == 'Digits':
                print('This takes a few minutes (i.e., around 2.5 minutes with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            X_ld_mssne = fmsnepy.mssne(X_hds=X_hds, n_components=dim_LDS, init=init_ld_emb, rand_state=fmsnepy.np.random.RandomState(2))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

            ###
            ###
            ###
            print('- Evaluating the DR quality of the LD embedding obtained using multi-scale SNE')
            t0 = time.time()
            rnx_mssne, auc_mssne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_mssne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_mssne, 4)))

            # Updating the lists for viz_qa
            L_rnx.append(rnx_mssne)
            Lmarkers.append('x')
            Lcols.append('red')
            Lleg_rnx.append('Ms SNE')
            Lls.append('solid')
            Lmedw.append(0.5)
            Lsdots.append(10)

            ###
            ###
            ###
            if plot_emb:
                print('- Plotting the LD embedding obtained using multi-scale SNE')
                print('If a figure is shown, close it to continue.')
                # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
                fmsnepy.viz_2d_emb(X=X_ld_mssne, vcol=labels, tit='LD embedding Ms SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
                print('===')
                print('===')
                print('===')

        ###
        ###
        ###
        # Applying t-SNE [7] if the data set is not too big, i.e. it is limited to a few thousands samples.
        if not big_data:
            print('- Applying t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
            if data_name == 'Digits':
                print('This takes a few minutes (i.e., around 2 minutes with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            X_ld_tsne = sklearn.manifold.TSNE(n_components=dim_LDS, perplexity=50.0, early_exaggeration=4.0, n_iter=1000, learning_rate=100.0, min_grad_norm=10.0**(-5), random_state=fmsnepy.np.random.RandomState(2), metric='euclidean', init=init_ld_emb, method='exact').fit_transform(X_hds)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

            ###
            ###
            ###
            print('- Evaluating the DR quality of the LD embedding obtained using t-SNE')
            t0 = time.time()
            rnx_tsne, auc_tsne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_tsne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_tsne, 4)))

            # Updating the lists for viz_qa
            L_rnx.append(rnx_tsne)
            Lmarkers.append('|')
            Lcols.append('black')
            Lleg_rnx.append('$t$-SNE')
            Lls.append('solid')
            Lmedw.append(0.5)
            Lsdots.append(10)

            ###
            ###
            ###
            if plot_emb:
                print('- Plotting the LD embedding obtained using t-SNE')
                print('If a figure is shown, close it to continue.')
                # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
                fmsnepy.viz_2d_emb(X=X_ld_tsne, vcol=labels, tit='LD embedding $t$-SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
                print('===')
                print('===')
                print('===')

        ###
        ###
        ###
        # Fast multi-scale t-SNE can be employed on very large-scale databases.
        print('- Applying fast multi-scale t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
        if data_name == 'Blobs':
            print('This takes a few seconds (i.e., around 32 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        elif data_name == 'Digits':
            print('This takes a few seconds (i.e., around 3 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        t0 = time.time()
        X_ld_fmstsne = fmsnepy.fmstsne(X_hds=X_hds, n_components=dim_LDS, init=init_ld_emb, rand_state=fmsnepy.np.random.RandomState(2), bht=0.75, fseed=1)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

        ###
        ###
        ###
        if big_data:
            print('- Evaluating the reduced DR quality of the LD embedding obtained using fast multi-scale t-SNE')
            if data_name == 'Blobs':
                print('This takes a few seconds (i.e., around 34 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            rnx_fmstsne, auc_fmstsne = fmsnepy.red_rnx_auc(X_hds=X_hds, X_lds=X_ld_fmstsne, Kup=Kup)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_fmstsne, 4)))
        else:
            print('- Evaluating the DR quality of the LD embedding obtained using fast multi-scale t-SNE')
            t0 = time.time()
            rnx_fmstsne, auc_fmstsne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_fmstsne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_fmstsne, 4)))

        # Updating the lists for viz_qa
        L_rnx.append(rnx_fmstsne)
        Lmarkers.append('s')
        Lcols.append('cyan')
        Lleg_rnx.append('FMs $t$-SNE')
        Lls.append('solid')
        Lmedw.append(0.5)
        Lsdots.append(10)

        ###
        ###
        ###
        if plot_emb:
            print('- Plotting the LD embedding obtained using fast multi-scale t-SNE')
            print('If a figure is shown, close it to continue.')
            # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
            fmsnepy.viz_2d_emb(X=X_ld_fmstsne, vcol=labels, tit='LD embedding FMs $t$-SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
            print('===')
            print('===')
            print('===')

        ###
        ###
        ###
        # Fast multi-scale SNE can be employed on very large-scale databases.
        print('- Applying fast multi-scale SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
        if data_name == 'Blobs':
            print('This takes a few minutes (i.e., around 15 minutes with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        elif data_name == 'Digits':
            print('This takes a few minutes (i.e., around 1.25 minutes with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        t0 = time.time()
        X_ld_fmssne = fmsnepy.fmssne(X_hds=X_hds, n_components=dim_LDS, init=init_ld_emb, rand_state=fmsnepy.np.random.RandomState(2), bht=0.45, fseed=1)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

        ###
        ###
        ###
        if big_data:
            print('- Evaluating the reduced DR quality of the LD embedding obtained using fast multi-scale SNE')
            if data_name == 'Blobs':
                print('This takes a few seconds (i.e., around 33 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            rnx_fmssne, auc_fmssne = fmsnepy.red_rnx_auc(X_hds=X_hds, X_lds=X_ld_fmssne, Kup=Kup)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_fmssne, 4)))
        else:
            print('- Evaluating the DR quality of the LD embedding obtained using fast multi-scale SNE')
            t0 = time.time()
            rnx_fmssne, auc_fmssne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_fmssne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_fmssne, 4)))

        # Updating the lists for viz_qa
        L_rnx.append(rnx_fmssne)
        Lmarkers.append('$\\star$')
        Lcols.append('magenta')
        Lleg_rnx.append('FMs SNE')
        Lls.append('solid')
        Lmedw.append(0.5)
        Lsdots.append(10)

        ###
        ###
        ###
        if plot_emb:
            print('- Plotting the LD embedding obtained using fast multi-scale SNE')
            print('If a figure is shown, close it to continue.')
            # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
            fmsnepy.viz_2d_emb(X=X_ld_fmssne, vcol=labels, tit='LD embedding FMs SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
            print('===')
            print('===')
            print('===')

        ###
        ###
        ###
        # Barnes-Hut (BH) t-SNE [8] can be employed on very large-scale databases.
        print('- Applying Barnes-Hut (BH) t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
        if data_name == 'Blobs':
            print('This takes a few minutes (i.e., around 5 minutes with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        elif data_name == 'Digits':
            print('This takes a few seconds (i.e., around 39 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        t0 = time.time()
        X_ld_bhtsne = sklearn.manifold.TSNE(n_components=dim_LDS, perplexity=50.0, early_exaggeration=12.0, n_iter=1000, learning_rate=200.0, min_grad_norm=10.0**(-5), random_state=fmsnepy.np.random.RandomState(2), metric='euclidean', init=init_ld_emb, method='barnes_hut', angle=0.5).fit_transform(X_hds)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))

        ###
        ###
        ###
        if big_data:
            print('- Evaluating the reduced DR quality of the LD embedding obtained using BH t-SNE')
            if data_name == 'Blobs':
                print('This takes a few seconds (i.e., around 33 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            rnx_bhtsne, auc_bhtsne = fmsnepy.red_rnx_auc(X_hds=X_hds, X_lds=X_ld_bhtsne, Kup=Kup)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_bhtsne, 4)))
        else:
            print('- Evaluating the DR quality of the LD embedding obtained using BH t-SNE')
            t0 = time.time()
            rnx_bhtsne, auc_bhtsne = fmsnepy.eval_dr_quality(d_hd=dm_hd, d_ld=fmsnepy.eucl_dist_matr(X_ld_bhtsne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=fmsnepy.rstr(t)))
            print('AUC: {v}'.format(v=fmsnepy.rstr(auc_bhtsne, 4)))

        # Updating the lists for viz_qa
        L_rnx.append(rnx_bhtsne)
        Lmarkers.append('o')
        Lcols.append('green')
        Lleg_rnx.append('BH $t$-SNE')
        Lls.append('solid')
        Lmedw.append(0.5)
        Lsdots.append(10)

        ###
        ###
        ###
        if plot_emb:
            print('- Plotting the LD embedding obtained using BH t-SNE')
            print('If a figure is shown, close it to continue.')
            # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
            fmsnepy.viz_2d_emb(X=X_ld_bhtsne, vcol=labels, tit='LD embedding BH $t$-SNE ({data_name} data set)'.format(data_name=data_name), fname=None, f_format=None)
            print('===')
            print('===')
            print('===')

        ###
        ###
        ###
        print('- Plotting the results of the DR quality assessment')
        print('If a figure is shown, close it to continue.')
        # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information.
        fmsnepy.viz_qa(Ly=L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel='Neighborhood size $K$', ylabel='$R_{\\mathrm{{NX}}}(K)$', fname=None, f_format=None, ncol_leg=2)
        print('===')
        print('===')
        print('===')

    ###
    ###
    ###
    print('*********************')
    print('***** Done! :-) *****')
    print('*********************')
