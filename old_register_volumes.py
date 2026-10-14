

    def register_volumes(self,target_volumes_list,show_plots=False,error_rms=0):

        for vidx,target_volume in enumerate(target_volumes_list):
            corr_arr = []
            y_shift_arr = []
            z_shift_arr = []
            x_shift_arr = []
            for s in range(self.ref.n_slow):
                tar = target_volume[s,:,:]
                res = self.ref.register(tar)

                if error_rms:
                    res['dx'] = res['dx'] + int(np.round(np.random.randn()*error_rms))
                    res['dy'] = res['dy'] + int(np.round(np.random.randn()*error_rms))
                    res['dz'] = res['dz'] + int(np.round(np.random.randn()*error_rms))
                
                corr_arr.append(res['xc'])
                x_shift_arr.append(res['dx'])
                y_shift_arr.append(res['dy']-s)
                z_shift_arr.append(res['dz'])

            if PLOT_SINGLE_VOLUME_RESULTS:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(corr_arr)
                plt.ylabel('xcorr')
                plt.subplot(2,2,2)
                plt.plot(x_shift_arr)
                plt.ylabel('x shift')
                plt.subplot(2,2,3)
                plt.plot(y_shift_arr)
                plt.ylabel('y shift')
                plt.subplot(2,2,4)
                plt.plot(z_shift_arr)
                plt.ylabel('z shift')
                plt.show()
                
            # filter shifts using xc here, later

            # ideas for filtering the shift vectors
            # 1. Limit derivative of shift to small amount; motivated
            #    by intuition that the retina's motion between B-scans
            #    is small, esp. at 5 kHz in AO-OCT system
            # 2. Median filtering may remove outliers when they are
            #    isolated, but will fail when clusters of B-scans are out
            #    of place, which appears to happen fequently (during saccades,
            #    for instance)
            # 3. Motion is a non-Markov process; the previous derivative (velocity)
            #    is predictive of the current derivative. This is a stronger statement
            #    than that made in #1 above.
            # 4. Eye movement processes are reversible; that is "previous" in #3 is
            #    equivalent to "next".

            corr_arr = np.array(corr_arr)
            y_shift_arr = np.array(y_shift_arr,dtype=float)
            z_shift_arr = np.array(z_shift_arr,dtype=float)
            x_shift_arr = np.array(x_shift_arr,dtype=float)
            rad_arr = np.sqrt(y_shift_arr**2+z_shift_arr**2+x_shift_arr**2)


            cstd = np.std(corr_arr)
            if cstd>0:
                ncorr_arr = corr_arr/np.std(corr_arr)
            else:
                ncorr_arr = 3.0*np.ones(corr_arr.shape)

            nrad_arr = rad_arr/(np.std(rad_arr)+1)

            ncorr_thresh_std = 1.5
            nrad_thresh_std = 1.0
            valid = (ncorr_arr>ncorr_thresh_std)*(nrad_arr<nrad_thresh_std)
            invalid = 1-valid
            
            corr_arr[np.where(invalid)] = np.nan
            y_shift_arr[np.where(invalid)] = np.nan
            z_shift_arr[np.where(invalid)] = np.nan
            x_shift_arr[np.where(invalid)] = np.nan

            if show_plots:
                plt.cla()
                plt.plot(ncorr_arr,label='normalized correlation')
                plt.plot(nrad_arr,label='normalized 3D displacement')
                for k in range(len(invalid)):
                    if invalid[k]:
                        plt.axvline(k,color='r',alpha=0.2)
                plt.legend()
                os.makedirs('figures',exist_ok=True)
                plt.savefig(os.path.join('figures','regdata_%03d.png'%vidx))
                plt.pause(.1)
            self.add(target_volume,x_shift_arr,y_shift_arr,z_shift_arr,corr_arr)

        self.correct_volumes()
        self.average_volume = np.nanmean(np.abs(np.array(self.corrected_volumes)),axis=0)
        
