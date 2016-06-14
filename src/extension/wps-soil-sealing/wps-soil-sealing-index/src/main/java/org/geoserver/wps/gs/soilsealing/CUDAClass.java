package org.geoserver.wps.gs.soilsealing;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemAllocHost;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.runtime.JCuda.cudaMemset;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferDouble;
import java.awt.image.DataBufferInt;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
//import java.util.Date;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import org.geoserver.wps.gs.CoverageImporter;
import org.geoserver.wps.gs.soilsealing.UrbanGridCUDAProcess.CUDABean;
import org.geotools.coverage.Category;
import org.geotools.coverage.GridSampleDimension;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridCoverageFactory;
import org.geotools.coverage.grid.io.AbstractGridFormat;
import org.geotools.factory.GeoTools;
import org.geotools.factory.Hints;
import org.geotools.gce.geotiff.GeoTiffFormat;
import org.geotools.gce.geotiff.GeoTiffWriter;
import org.geotools.process.ProcessException;
import org.opengis.coverage.grid.GridCoverage;
import org.opengis.parameter.GeneralParameterValue;
import org.opengis.parameter.ParameterValueGroup;
import org.opengis.util.InternationalString;

import com.sun.media.imageioimpl.common.ImageUtil;

public class CUDAClass {

    /** Logger used for logging exceptions */
    public static final Logger LOGGER = Logger.getLogger(CUDAClass.class.toString());

    private final static String PTXFILE_fragmentation = "/opt/soil_sealing/cudacodes/fragmentation.ptx";

    private final static String PTXFILE_land_take = "/opt/soil_sealing/cudacodes/land_take.ptx";

    private final static String PTXFILE_perimeter = "/opt/soil_sealing/cudacodes/perimeter.ptx";

    private final static String PTXFILE_ccl_1toN_hist = "/opt/soil_sealing/cudacodes/ccl.ptx";

    // private Clock start_t,end_t;

    // 1 squared meter is m2_to_ha hectares
    private final static double m2_to_ha = 1D / 10000D;

    // 6 persons/year are fed by 1 hectare of harvested land cultivated with wheat
    private final static double fedPersons = 6.0;// [persons * year-1 * ha-1]

    private final static int numberOfBins = 4;

    private final static int BLOCKDIM_X = 32;

    private final static int BLOCKDIM = 256;

    // ***perimeter :: _pi***
    // [reduce6] No of threads working in single block
    private final static int threads_pi = 512;

    // [reduce6] No of blocks working in grid (this gives also the size of
    // output Perimeter, to be summed outside CUDA)
    private final static int blocks_pi = 64;

    // [tidx2_ns] No of pixels processed by single thread
    private final static int mask_len_pi = 40;

    private final static int BLOCKDIM_X_ccl = 32;

    private final static int threads_ccl = 512;

    // *** land take***
    private final static int threads_lt = 512;

    /*
     * public static void CUDA_CHECK_RETURN(int value){ if ( value != 0 ) { System.out.println(stderr, "Error %s at line %d in file %s\n",
     * cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); exit(1); } }
     */
    /*
     * #define CUDA_CHECK_RETURN(err) __checkCudaErrors (err, __FILE__, __LINE__)
     * 
     * inline void __checkCudaErrors( CUresult err, const char *file, const int line ) { if( CUDA_SUCCESS != err) { fprintf(stderr,
     * "CUDA Driver API error = %04d from file <%s>, line %i.\n", err, file, line ); exit(-1); } }
     */

    /**
     * 
     * @param beans
     * @param areaOfOnePixel
     * @param year
     * @param admin_unit
     * @param Distribution : from 1 to 100, sets the percentage of sorted bin sizes to retain whose range is [1, Distribution] and disregards the
     *        range of bin sizes ]Distribution,100].
     * @return
     * @throws FileNotFoundException
     */
    public double urban_sprawl(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit, int Distribution) throws FileNotFoundException {
        // urban sprawl = SUD / SUT;
        // SUD = ccl_1toN_hist + sort;
        // SUT = reduce6(BIN*ROI) * cellsize;

        int SUD = 0;

        // **SUD**
        Map<String, Object> result = ccl_1toN_hist(beans, year, admin_unit);
        // **SUT**
        int SUTval = SUT(beans, year, admin_unit);

        // sort function: Java or Cuda?
        int[] sortedArray = ((int[]) result.get("h_histogram")).clone();
        int N_polygons = ((int) result.get("Nbins_0")) - 1;

        Arrays.sort(sortedArray);// !!ASCENDING!!
        for (int ii = 1; ii <= (int) (N_polygons * (Distribution / 100.0)); ii++) {
            SUD += sortedArray[ii];
        }

        double I_05_urbanSprawl = ((double) SUD / (double) SUTval) * 100;
        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_05_urbanSprawl, "ssgci__05");
            storeIndex_double((double) SUD, "ssgci__SUD");
            storeIndex_double((double) SUTval, "ssgci__SUT");
        }
        return I_05_urbanSprawl;
    }

    public static double edge_density(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit) throws IOException {
        // edge density = ∑Perimeter / ST;
        // ∑Perimeter = perimeter;
        // ST = map_len * cellsize;

        double cellSize = beans.get(0).getCellSize();
        // int WIDTH = beans.get(admin_unit).width;
        // int HEIGHT = beans.get(admin_unit).height;

        // ∑Perimeter
        int Sperimeter = CUDAClass.perimeter(beans, year, admin_unit);
        // **SUT**
        int SUTval = SUT(beans, year, admin_unit);

        double I_06_edgeDensity = ((double) Sperimeter * cellSize) / (SUTval * areaOfOnePixel)
                / m2_to_ha;
        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_06_edgeDensity, "ssgci__06");
            storeIndex_double((double) SUTval, "ssgci__SUT");
            storeIndex_double((double) Sperimeter, "ssgci__Sperimeter");
        }

        return I_06_edgeDensity;
    }

    public static double urban_area(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit) throws FileNotFoundException {
        // urban area = SUT / ST;
        // SUT = reduce6(BIN*ROI) * cellsize;
        // ST = map_len * cellsize;
        int SUTval = 0;
        int WIDTH = beans.get(admin_unit).width;
        int HEIGHT = beans.get(admin_unit).height;
        int map_len = WIDTH * HEIGHT;

        // **SUT**
        SUTval = SUT(beans, year, admin_unit);
        // **ST**
        double ST = map_len;

        double I_07a_UrbanArea = ((double) SUTval / ST) * 100;

        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_07a_UrbanArea, "ssgci__07a");
            storeIndex_double((double) SUTval, "ssgci__SUT");
            storeIndex_double(ST, "ssgci__ST");
        }

        return I_07a_UrbanArea;
    }

    public double highest_polygon_ratio(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit) throws FileNotFoundException {
        // highest polygon ratio = S_poly_max / SUT;
        // S_poly_max = ccl_1toN_hist + max(take);
        // SUT = reduce6(BIN*ROI) * cellsize;

        // **S_poly_max**
        Map<String, Object> result = ccl_1toN_hist(beans, year, admin_unit);
        // **SUT**
        int SUTval = SUT(beans, year, admin_unit);

        int[] sortedArray = ((int[]) result.get("h_histogram")).clone();
        Arrays.sort(sortedArray);// !!ASCENDING!!
        double maxHistogramValue = (double) sortedArray[sortedArray.length - 1];

        // areaOfOnePixel is useless because it's both at numerator and denominator
        double I_07b_HighestPolygonRatio = (maxHistogramValue / (double) SUTval) * 100.0;

        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_07b_HighestPolygonRatio, "ssgci__07b");
            storeIndex_double((double) SUTval, "ssgci__SUT");
            storeIndex_double(maxHistogramValue, "ssgci__maxHistogramValue");
        }
        return I_07b_HighestPolygonRatio;
    }

    public double others_polygons_avesurf(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit) throws FileNotFoundException {
        // others polygons avesurf = S_poly_~max / N_poly_~max;
        // S_poly_~max = ccl_1toN_hist + max(delete)
        // N_poly_~max = Nbins -1;

        int ii = 0;
        double S_poly_notmax = 0;

        // **S_poly_~max**
        Map<String, Object> result = ccl_1toN_hist(beans, year, admin_unit);

        // **N_poly_~max**
        int N_other_polys = (int) result.get("Nbins_0") - 2;
        int[] sortedArray = ((int[]) result.get("h_histogram")).clone();

        Arrays.sort(sortedArray);// !!ASCENDING!!
        double maxHistogramValue = sortedArray[sortedArray.length - 1];
        for (ii = 1; ii < N_other_polys + 2; ii++) {
            S_poly_notmax += (double) sortedArray[ii];
        }
        S_poly_notmax = (S_poly_notmax - maxHistogramValue) * areaOfOnePixel;

        double I_07c_OthersPolygonRatio = (S_poly_notmax / (double) N_other_polys) * m2_to_ha;

        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_07c_OthersPolygonRatio, "ssgci__07c");
            storeIndex_double(S_poly_notmax, "ssgci__S_poly_notmax");
            storeIndex_double(maxHistogramValue, "ssgci__maxHistogramValue");
        }

        return I_07c_OthersPolygonRatio;
    }

    public double[] modelUrbanDevelopment(List<CUDABean> beans, double areaOfOnePixel, int year,
            int admin_unit) throws IOException {

        int ii = 0;
        double S_poly_notmax = 0;
        double cellSize = beans.get(0).getCellSize();
        // int WIDTH = beans.get(admin_unit).width;
        // int HEIGHT = beans.get(admin_unit).height;
        double mUD[] = { 0, 0, 0 };
        int SUTval = 0;

        // **CCL/HIST**
        Map<String, Object> result = ccl_1toN_hist(beans, year, admin_unit);
        // **SUT**
        SUTval = SUT(beans, year, admin_unit);
        // ∑Perimeter
        int Sperimeter = CUDAClass.perimeter(beans, year, admin_unit);

        int N_other_polys = (int) result.get("Nbins_0") - 2;
        int[] sortedArray = ((int[]) result.get("h_histogram")).clone();

        Arrays.sort(sortedArray);// !!ASCENDING!!
        double maxHistogramValue = (double) sortedArray[sortedArray.length - 1];
        for (ii = 1; ii < N_other_polys + 2; ii++) {
            S_poly_notmax += (double) sortedArray[ii];
        }
        S_poly_notmax = (S_poly_notmax - maxHistogramValue) * areaOfOnePixel;

        // 01 **LCPI | highest polygon ratio**
        double I_07b_HighestPolygonRatio = (maxHistogramValue / (double) SUTval) * 100.0; // *
                                                                                          // areaOfOnePixel
                                                                                          // is
                                                                                          // useless
                                                                                          // because
                                                                                          // it's
                                                                                          // both
                                                                                          // at
                                                                                          // numerator
                                                                                          // and
                                                                                          // denominator

        // 02 **RMPS | others polygon ratio**
        double I_07c_OthersPolygonRatio = (S_poly_notmax / (double) N_other_polys) * m2_to_ha;

        // 03 **EDclass | edge density**
        double I_06_edgeDensity = (((double) Sperimeter * cellSize) / (SUTval * areaOfOnePixel))
                / m2_to_ha;

        if (UrbanGridCUDAProcess.TESTING) {
            storeIndex_double(I_07b_HighestPolygonRatio, "ssgci__07b");
            storeIndex_double(I_07c_OthersPolygonRatio, "ssgci__07c");
            storeIndex_double(I_06_edgeDensity, "ssgci__06");
            storeIndex_double(S_poly_notmax, "ssgci__S_poly_notmax");
            storeIndex_double(maxHistogramValue, "ssgci__maxHistogramValue");
            storeIndex_double((double) SUTval, "ssgci__SUT");
            storeIndex_double((double) Sperimeter, "ssgci__Sperimeter");
        }

        mUD[0] = I_07b_HighestPolygonRatio;
        mUD[1] = I_07c_OthersPolygonRatio;
        mUD[2] = I_06_edgeDensity;

        return mUD;
    }

    /**
     * 
     * @param beans :: list of layers/stuff and their properties
     * @param year :: index defining the year in which calculate perimeter
     * @param admin_unit :: index defining the administrative unit which is undergoing calculation
     * @return :: SUT which is the sum of BINf (=BIN*ROI)
     */
    public static int SUT(List<CUDABean> beans, int year, int admin_unit) {
        /**
         * NOTES Now the algorithm works fine both in CUDA-C and in JCuda versions
         */

        /*
         * PARAMETERS
         */
        CUresult err;
        CUcontext context = null;
        CUmodule module = null;

        CUdeviceptr dev_BIN = null;
        CUdeviceptr dev_ROI = null;
        CUdeviceptr dev_BINf = null;
        CUdeviceptr d_SUT = null;

        // int gpuDeviceCount[] = { 0 };
        int elapsed_time = 0;
        // count the number of kernels that must print their output:
        int count_print = 0;
        int ii = 0;
        int SUT = 0;

        try {
            int deviceCount = getGPUDeviceCount();

            /*
             * SELECT THE FIRST DEVICE (but I should select/split devices according to streams)
             */
            int selDev = getRandomGpuDevice(deviceCount);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, selDev);
            // Query some useful properties:
            int amountProperty[] = { 0 };
            // -1-
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            int maxThreadsPerBlock = amountProperty[0];
            // -2-
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
            int MaxGridDimX = amountProperty[0];
            // -3-
            // ...others?

            /*
             * CREATE THE CONTEXT (for currently selected device)
             */
            context = new CUcontext();
            // int cuCtxCreate_STATUS =
            cuCtxCreate(context, selDev, device);
            // Load the ptx file:
            module = new CUmodule();

            cuModuleLoad(module, PTXFILE_perimeter);
            /*
             * ENTRY POINTS (obtain function pointers to the entry kernels) _Z14reduce6_doublePKdPKhPdjjb
             */

            // BIN x ROI
            CUfunction F_bin_x_roi = new CUfunction();
            cuModuleGetFunction(F_bin_x_roi, module, "_Z9bin_x_roiPKhS0_Phjj");
            // reduce6_double
            CUfunction F_reduce6_char = new CUfunction();
            cuModuleGetFunction(F_reduce6_char, module, "_Z12reduce6_binfPKhPjjjb");

            /*
             * DIM of ARRAYS in BYTES
             */
            int WIDTH = beans.get(admin_unit).width;
            int HEIGHT = beans.get(admin_unit).height;
            int map_len = WIDTH * HEIGHT;
            int sizeChar = map_len * Sizeof.BYTE;
            // long sizeDouble = map_len*Sizeof.DOUBLE;

            /*
             * CPU ARRAYS
             */
            int h_SUT[] = new int[blocks_pi];
            cuMemAllocHost(Pointer.to(h_SUT), blocks_pi * Sizeof.INT);
            /*
             * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
             */
            // get pointers
            dev_BIN = new CUdeviceptr();
            dev_ROI = new CUdeviceptr();
            dev_BINf = new CUdeviceptr();
            d_SUT = new CUdeviceptr();

            // allocate in mem
            cuMemAlloc(dev_BIN, sizeChar);
            cuMemAlloc(dev_ROI, sizeChar);
            cuMemAlloc(dev_BINf, sizeChar);
            cuMemAlloc(d_SUT, blocks_pi * Sizeof.INT);

            /*
             * MEM COPY H2D
             */
            if (year == 0) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getReferenceImage()),
                        sizeChar);
            } else if (year == 1) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getCurrentImage()), sizeChar);
            }
            cuMemcpyHtoD(dev_ROI, Pointer.to(beans.get(admin_unit).roi), sizeChar);

            int smemSize = /* (threads_pi <= 32) ? 2 * threads_pi * Sizeof.INT : */threads_pi
                    * Sizeof.INT;

            // ***00***
            // bin_x_roi<<< dimGrid, dimBlock >>>( dev_BIN, dev_ROI, dev_BINf,
            // map_len, 512 );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_bin_x_roi = Pointer.to(Pointer.to(dev_BIN), Pointer.to(dev_ROI),
                    Pointer.to(dev_BINf), Pointer.to(new int[] { map_len }),
                    Pointer.to(new int[] { threads_pi }));
            // Call the kernel function.
            cuLaunchKernel(F_bin_x_roi, blocks_pi, 1, 1, // Grid dimension :: dim3
                                                         // dimGrid( blocks, 1, 1
                                                         // );
                    threads_pi, 1, 1, // Block dimension :: dim3 dimBlock( threads,
                                      // 1, 1 );
                    0, null, // Shared memory size and stream
                    kern_bin_x_roi, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();

            // ***01***
            // reduce6_binf<<< dimGrid, dimBlock, smemSize >>>( dev_BINf, d_SUT,
            // map_len, 512, isPow2(map_len) );
            // start_t = clock();
            byte nIsPow2 = (byte) (((map_len & (map_len - 1)) == 0) ? 1 : 0);
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_reduce6 = Pointer.to(Pointer.to(dev_BINf), Pointer.to(d_SUT),
                    Pointer.to(new int[] { map_len }), Pointer.to(new int[] { threads_pi }),
                    Pointer.to(new byte[] { nIsPow2 }));
            // Call the kernel function.
            cuLaunchKernel(F_reduce6_char, blocks_pi, 1, 1, // Grid dimension ::
                                                            // dim3 dimGrid( blocks,
                                                            // 1, 1 );
                    threads_pi, 1, 1, // Block dimension :: dim3 dimBlock( threads,
                                      // 1, 1 );
                    smemSize, null, // Shared memory size and stream
                    kern_reduce6, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_24,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            /*
             * if(UrbanGridCUDAProcess.TESTING){ byte BIN[] = new byte[map_len]; byte ROI[] = new byte[map_len]; if (year == 0) { BIN =
             * beans.get(admin_unit).getReferenceImage(); }else if(year == 1) { BIN = beans.get(admin_unit).getCurrentImage(); }
             * 
             * ROI = beans.get(admin_unit).roi; int SUT_cpu = 0; for(ii=0;ii<map_len;ii++){ SUT_cpu += (int) (BIN[ii]*ROI[ii]); } //ALTERNATIVE :: sum
             * = IntStream.of(a).parallel().sum(); }
             */

            // extract perimeter:
            cuMemcpyDtoH(Pointer.to(h_SUT), d_SUT, blocks_pi * Sizeof.INT);
            for (ii = 0; ii < blocks_pi; ii++) {
                SUT += h_SUT[ii];
            }

            return SUT;
        } finally {
            try {
                // FREE MEMORY:
                // ...on the host:
                // It seems that host mem cannot be freed!
                // cudaFreeHost(lab_mat_cpu);
                // ...on the device:
                if (dev_BIN != null)
                    cuMemFree(dev_BIN);
                if (dev_BINf != null)
                    cuMemFree(dev_BINf);
                if (dev_ROI != null)
                    cuMemFree(dev_ROI);
                if (d_SUT != null)
                    cuMemFree(d_SUT);

                // Unload MODULE
                if (module != null)
                    cuModuleUnload(module);

                // Destroy CUDA context:
                if (context != null)
                    cuCtxDestroy(context);
            } catch (Exception e) {
                // LOG THE EXCEPTION ...
            }
        }
    }

    /**
     * 
     * @param beans :: list of layers/stuff and their properties
     * @param year :: index defining the year in which calculate perimeter (reference, current)
     * @param admin_unit :: index defining the administrative unit which is undergoing calculation
     * @return :: a list with three objects: (1) a histogram counts, (2) No. of bins, and (3) a map of labels.
     */
    private Map<String, Object> ccl_1toN_hist(List<CUDABean> beans, int year, int admin_unit) {
        /**
         * Copyright 2016 Giuliano Langella: ---- check Jcuda version!! ----
         * 
         * This function computes the perimeter in driver-API CUDA-C. ************************* -0- filter_roi -1- intra_tile_labeling | ––> 1st Stage
         * :: intra-tile :: mandatory ––––| | -2- stitching_tiles |\ | *random CCL labeling* -3- root_equivalence |_|––> 2nd Stage :: inter-tiles ::
         * mandatory | | -4- intra_tile_re_label | ––> 3rd Stage :: intra-tile :: mandatory ––––|
         * 
         * -5- count_labels |\ -6- labels__1_to_N | |––> 4th Stage :: labels 1 to N :: mandatory #2 ––––| -7- intratile_relabel_1toN |/ | | *size of
         * each label* -8- del_duplicated_lines | ––> 5th Stage :: adjust size :: mandatory #2 | | -9- histogram(_shmem) | ––> 6th Stage :: histogram
         * :: mandatory #2 ––––| *************************
         * 
         * 
         * Object: Raster-scan and label-equivalence-based algorithm. Authors: Giuliano Langella email: gyuliano@libero.it
         * 
         * ----------- DESCRIPTION: -----------
         * 
         * I: "urban" --> [0,0] shifted O: "lab_mat" --> [1,1] shifted
         * 
         * The "forward scan mask" for eight connected connectivity is the following: nw nn ne ww cc xx xx xx xx assuming that: > cc is the
         * background(=0)/foreground(=1) pixel at (r,c), > nw, nn, ne, ww are the north-west, north, north-east and west pixels in the eight connected
         * connectivity, > xx are skipped pixels. Therefore the mask has 4 active pixels with(out) object pixels (that is foreground pixels).
         */

        synchronized (this) {
            /*
             * PARAMETERS
             */
            CUresult err;
            CUcontext context = null;
            CUmodule module = null;
            /*
             * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
             */
            // get pointers
            CUdeviceptr urban_gpu = null;
            CUdeviceptr d_BINf = null;
            CUdeviceptr dev_ROI = null;
            CUdeviceptr lab_mat_gpu = null;
            CUdeviceptr lab_mat_gpu_1N = null;
            CUdeviceptr lab_mat_gpu_f = null;
            CUdeviceptr bins_gpu = null;
            CUdeviceptr d_histogram = null;

            // int gpuDeviceCount[] = { 0 };
            int elapsed_time = 0;
            // count the number of kernels that must print their output:
            int count_print = 0;
            // floor(sqrt(devProp.maxThreadsPerBlock));
            int sqrt_nmax_threads = BLOCKDIM_X_ccl;

            try {
                int deviceCount = getGPUDeviceCount();

                /*
                 * SELECT THE FIRST DEVICE (but I should select/split devices according to streams)
                 */
                int selDev = getRandomGpuDevice(deviceCount);

                System.out.println(" - GPU Device: [" + (selDev + 1) + "] / [" + deviceCount + "]");

                CUdevice device = new CUdevice();
                cuDeviceGet(device, selDev);
                // Query some useful properties:
                int amountProperty[] = { 0 };
                // -1-
                cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                        device);
                int maxThreadsPerBlock = amountProperty[0];
                // -2-
                cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
                int MaxGridDimX = amountProperty[0];
                // -3-
                cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        device);
                int N_sm = amountProperty[0];
                // -4-
                cuDeviceGetAttribute(amountProperty,
                        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
                int max_threads_per_SM = amountProperty[0];
                // -5-
                cuDeviceGetAttribute(amountProperty,
                        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
                int sharedMemPerBlock = amountProperty[0];

                /*
                 * CREATE THE CONTEXT (for currently selected device)
                 */
                context = new CUcontext();
                // int cuCtxCreate_STATUS =
                cuCtxCreate(context, selDev, device);
                // Load the ptx file:
                module = new CUmodule();
                cuModuleLoad(module, PTXFILE_ccl_1toN_hist);// run grep -in entry
                // ccl.ptx
                /*
                 * ENTRY POINTS (obtain function pointers to the entry kernels) -0- _Z10filter_roiPhPKhj -1- _Z19intra_tile_labelingPKhjjjjPj -2-
                 * _Z15stitching_tilesPjjjj -3- _Z16root_equivalencePjjjj -4- _Z19intra_tile_re_labeljjPj -5- _Z12count_labelsjjPKjPj -6-
                 * _Z14labels__1_to_NjjPKjPjS1_jjS1_S1_ -7- _Z22intratile_relabel_1toNjjPKjPj -8- _Z20del_duplicated_linesPKjjjPjjj -9-
                 * _Z15histogram_shmemPKjPKhPjjj -9- _Z9histogramPKjPKhPjjj
                 */
                // -0-
                CUfunction F_filter_roi = new CUfunction();
                cuModuleGetFunction(F_filter_roi, module, "_Z10filter_roiPhPKhj");
                // -1-
                CUfunction F_intra_tile_labeling = new CUfunction();
                cuModuleGetFunction(F_intra_tile_labeling, module,
                        "_Z19intra_tile_labelingPKhjjjjPj");
                // -2-
                CUfunction F_stitching_tiles = new CUfunction();
                cuModuleGetFunction(F_stitching_tiles, module, "_Z15stitching_tilesPjjjj");
                // -3-
                CUfunction F_root_equivalence = new CUfunction();
                cuModuleGetFunction(F_root_equivalence, module, "_Z16root_equivalencePjjjj");
                // -4-
                CUfunction F_intra_tile_re_label = new CUfunction();
                cuModuleGetFunction(F_intra_tile_re_label, module, "_Z19intra_tile_re_labeljjPj");
                // -5-
                CUfunction F_count_labels = new CUfunction();
                cuModuleGetFunction(F_count_labels, module, "_Z12count_labelsjjPKjPj");
                // -6-
                CUfunction F_labels__1_to_N = new CUfunction();
                cuModuleGetFunction(F_labels__1_to_N, module,
                        "_Z14labels__1_to_NjjPKjPjS1_jjS1_S1_");
                // -7-
                CUfunction F_intratile_relabel_1toN = new CUfunction();
                cuModuleGetFunction(F_intratile_relabel_1toN, module,
                        "_Z22intratile_relabel_1toNjjPKjPj");
                // -8-
                CUfunction F_del_duplicated_lines = new CUfunction();
                cuModuleGetFunction(F_del_duplicated_lines, module,
                        "_Z20del_duplicated_linesPKjjjPjjj");
                // -9a-
                CUfunction F_histogram_shmem = new CUfunction();
                cuModuleGetFunction(F_histogram_shmem, module, "_Z15histogram_shmemPKjPKhPjjj");
                // -9b-
                CUfunction F_histogram = new CUfunction();
                cuModuleGetFunction(F_histogram, module, "_Z9histogramPKjPKhPjjj");

                /*
                 * ALLOCATION
                 */
                int WIDTH = beans.get(admin_unit).width;
                int HEIGHT = beans.get(admin_unit).height;
                int map_len = WIDTH * HEIGHT;
                int tiledimX = sqrt_nmax_threads;
                int tiledimY = sqrt_nmax_threads;
                // I need to add a first row of zeros, to let the kernels work fine also
                // at location (0,0),
                // where the LABEL cannot be different from zero (the LABEL is equal to
                // thread absolute position).
                int HEIGHT_1 = HEIGHT + 1;
                // X-dir of extended array
                int ntilesX = (int) Math.ceil((double) (WIDTH - 1) / (double) (tiledimX - 1));
                int ntX_less = (int) Math.floor((double) (WIDTH - 1) / (double) (tiledimX - 1));
                int WIDTH_e = (ntilesX - ntX_less) + (ntX_less * tiledimX)
                        + (WIDTH - 1 - ntX_less * (tiledimX - 1));
                // Y-dir of extended array
                int ntilesY = (int) Math.ceil((double) (HEIGHT_1 - 1) / (double) (tiledimY - 1));
                int ntY_less = (int) Math.floor((double) (HEIGHT_1 - 1) / (double) (tiledimY - 1));
                int HEIGHT_e = (ntilesY - ntY_less) + (ntY_less * tiledimY)
                        + (HEIGHT_1 - 1 - ntY_less * (tiledimY - 1));
                /* ....::: DIM of ARRAYS in BYTES :::.... */
                long sizeChar = WIDTH * HEIGHT * Sizeof.BYTE; // it does not need the
                // offset
                long sizeChar_o = WIDTH * HEIGHT_1 * Sizeof.BYTE; // it accounts for the
                // offset
                // long sizeUintL = WIDTH * HEIGHT_1 * sizeof.byte;
                long sizeUintL_s = WIDTH * HEIGHT * Sizeof.INT;
                long sizeUintL_e = WIDTH_e * HEIGHT_e * Sizeof.INT; // the offset is
                // considered (using
                // HEIGHT_1 to
                // define HEIGHT_e)
                long sizeBins = ntilesX * ntilesY * Sizeof.INT;

                /*
                 * CPU ARRAYS
                 */
                int lab_mat_cpu[] = new int[WIDTH_e * HEIGHT_e];
                int lab_mat_cpu_f[] = new int[WIDTH * HEIGHT];
                int bins_cpu[] = new int[ntilesX * ntilesY];
                int cumsum[] = new int[ntilesX * ntilesY];
                byte TMP[] = new byte[WIDTH * HEIGHT];
                byte urban_cpu[] = new byte[WIDTH * HEIGHT_1];
                byte h_binroi_1[] = new byte[WIDTH * HEIGHT_1];
                cuMemAllocHost(Pointer.to(lab_mat_cpu), sizeUintL_e);
                cuMemAllocHost(Pointer.to(lab_mat_cpu_f), sizeUintL_s);
                cuMemAllocHost(Pointer.to(bins_cpu), sizeBins);
                cuMemAllocHost(Pointer.to(cumsum), sizeBins);
                cuMemAllocHost(Pointer.to(TMP), sizeChar);
                cuMemAllocHost(Pointer.to(urban_cpu), sizeChar_o);
                cuMemAllocHost(Pointer.to(h_binroi_1), sizeChar_o);
                // Arrays.fill( TMP, (byte) 0 );

                /*
                 * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
                 */
                // get pointers
                urban_gpu = new CUdeviceptr();
                d_BINf = new CUdeviceptr();
                dev_ROI = new CUdeviceptr();
                lab_mat_gpu = new CUdeviceptr();
                lab_mat_gpu_1N = new CUdeviceptr();
                lab_mat_gpu_f = new CUdeviceptr();
                bins_gpu = new CUdeviceptr();
                // allocate in mem
                cuMemAlloc(urban_gpu, sizeChar_o);
                cuMemAlloc(d_BINf, sizeChar);
                cuMemAlloc(dev_ROI, sizeChar_o);
                cuMemAlloc(lab_mat_gpu, sizeUintL_e);
                cuMemAlloc(lab_mat_gpu_1N, sizeUintL_e);
                cuMemAlloc(lab_mat_gpu_f, sizeUintL_s);
                cuMemAlloc(bins_gpu, sizeBins);
                // set mem
                cudaMemset(urban_gpu, 0, sizeChar_o);
                cudaMemset(d_BINf, 0, sizeChar);
                cudaMemset(dev_ROI, 0, sizeChar_o);
                cudaMemset(bins_gpu, 0, sizeBins);
                cudaMemset(lab_mat_gpu, 0, sizeUintL_e);
                cudaMemset(lab_mat_gpu_1N, 0, sizeUintL_e);
                cudaMemset(lab_mat_gpu_f, 0, sizeUintL_s);
                /*
                 * MEM COPY H2D
                 */

                Arrays.fill(h_binroi_1, (byte) 0);

                if (year == 0) {
                    System.arraycopy(beans.get(admin_unit).getReferenceImage(), 0, h_binroi_1,
                            WIDTH, map_len);
                    cuMemcpyHtoD(urban_gpu, Pointer.to(h_binroi_1), sizeChar_o);
                } else if (year == 1) {
                    System.arraycopy(beans.get(admin_unit).getCurrentImage(), 0, h_binroi_1, WIDTH,
                            map_len);
                    cuMemcpyHtoD(urban_gpu, Pointer.to(h_binroi_1), sizeChar_o);
                } else if (year == 2) {
                    // do nothing because the DIFF is calculated in JAVA ==> make the
                    // JCuda/Cuda version to speedup computation!!

                    /*
                     * System.out.println( "You should implement in CUDA the difference between ref & curr!!" ); for(int ii=0;ii<map_len;ii++){
                     * host_PERI[ii] = beans.get(admin_unit ).referenceImage[ii]-beans.get(admin_unit).getCurrentImage()[ii]; } return host_PERI;
                     */}
                Arrays.fill(h_binroi_1, (byte) 0);
                System.arraycopy(beans.get(admin_unit).roi, 0, h_binroi_1, WIDTH, map_len);
                cuMemcpyHtoD(dev_ROI, Pointer.to(h_binroi_1), sizeChar_o);

                /*
                 * KERNELS GEOMETRY NOTE: use ceil() instead of the "%" operator!!!
                 */
                int sh_mem = (tiledimX * tiledimY) * (Sizeof.INT);
                char sh_mem_f = (threads_ccl) * (Sizeof.BYTE);

                int num_blocks_per_SM = max_threads_per_SM / threads_ccl;// e.g.
                // 1536/512
                // = 3
                int Nblks_per_grid = N_sm * num_blocks_per_SM;
                int mapel_per_thread = (int) Math.ceil((double) map_len
                        / (double) ((threads_ccl * 2) * Nblks_per_grid));// e.g. n /
                // (14*3*512*2)

                /*
                 * KERNELS INVOCATION
                 */
                // ***00***
                // filter_roi<<<dimGrid,dimBlock>>>(urban_gpu,dev_ROI,map_len);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_00 = Pointer.to(Pointer.to(urban_gpu), Pointer.to(dev_ROI),
                        Pointer.to(new int[] { WIDTH * HEIGHT_1 }));
                // Call the kernel function.
                cuLaunchKernel(F_filter_roi, Nblks_per_grid, 1, 1, // Grid dimension ::
                        // dimGrid(
                        // Nblks_per_grid,
                        // 1, 1 )
                        threads_ccl, 1, 1, // Block dimension :: dimBlock( threads_ccl,
                        // 1, 1 )
                        0, null, // Shared memory size and stream
                        kern_00, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                // ***01***
                // intra_tile_labeling<<<grid,block,sh_mem>>>(urban_gpu,WIDTH,HEIGHT_1,WIDTH_e,HEIGHT_e,lab_mat_gpu);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_01 = Pointer.to(Pointer.to(urban_gpu),
                        Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT_1 }),
                        Pointer.to(new int[] { WIDTH_e }), Pointer.to(new int[] { HEIGHT_e }),
                        Pointer.to(lab_mat_gpu));
                // Call the kernel function.
                cuLaunchKernel(F_intra_tile_labeling, ntilesX, ntilesY, 1, // Grid
                        // dimension
                        // ::
                        // grid(ntilesX,ntilesY,1)
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        sh_mem, null, // Shared memory size and stream
                        kern_01, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
                if (UrbanGridCUDAProcess.TESTING) {
                    try {
                        Pointer kern_08 = Pointer.to(Pointer.to(lab_mat_gpu),
                                Pointer.to(new int[] { WIDTH_e }),
                                Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu_f),
                                Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }));
                        // Call the kernel function.
                        cuLaunchKernel(F_del_duplicated_lines, ntilesX, ntilesY, 1, // Grid
                                // dimension
                                // ::
                                // grid(ntilesX,ntilesY,1);
                                tiledimX, tiledimY, 1, // Block dimension ::
                                // block(tiledimX,tiledimY,1);
                                0, null, // Shared memory size and stream
                                kern_08, null // Kernel- and extra parameters
                        );
                        cuCtxSynchronize();
                        // end_t = clock();
                        // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                        // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000
                        // ));
                        // elapsed_time += (int)( (double)(end_t - start_t ) /
                        // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                        cuMemcpyDtoH(Pointer.to(lab_mat_cpu_f), lab_mat_gpu_f, sizeUintL_s);

                        UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT,
                                lab_mat_cpu_f, DataBuffer.TYPE_INT, "ssgci__intra_tile_labeling");

                    } catch (IOException e) {
                        LOGGER.log(Level.WARNING, "Could not save GeoTIFF Sample for testing", e);
                    }
                }

                // ***02***
                // stitching_tiles<<<grid,block_2>>>(lab_mat_gpu,tiledimY, WIDTH_e,
                // HEIGHT_e);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_02 = Pointer.to(Pointer.to(lab_mat_gpu),
                        Pointer.to(new int[] { tiledimY }), Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }));
                // Call the kernel function.
                cuLaunchKernel(F_stitching_tiles, ntilesX, ntilesY, 1, // Grid dimension
                        // ::
                        // grid(ntilesX,ntilesY,1)
                        tiledimX, 1, 1, // Block dimension :: block_2(tiledimX,1,1); //
                        // ==> this is only possible if the block is
                        // squared !!!!!!!! Because I use the same
                        // threads for cols & rows
                        0, null, // Shared memory size and stream
                        kern_02, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                // ***03***
                // root_equivalence<<<grid_2,block_2>>>(lab_mat_gpu,tiledimY, WIDTH_e,
                // HEIGHT_e);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_03 = Pointer.to(Pointer.to(lab_mat_gpu),
                        Pointer.to(new int[] { tiledimY }), Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }));
                // Call the kernel function.
                cuLaunchKernel(F_root_equivalence, ntilesX, ntilesY, 1, // Grid
                        // dimension ::
                        // grid_2(ntilesX,ntilesY,1);
                        tiledimX, 1, 1, // Block dimension :: block_2(tiledimX,1,1); //
                        // ==> this is only possible if the block is
                        // squared !!!!!!!! Because I use the same
                        // threads for cols & rows
                        0, null, // Shared memory size and stream
                        kern_03, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                // ***04***
                // intra_tile_re_label<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_04 = Pointer.to(Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu));
                // Call the kernel function.
                cuLaunchKernel(F_intra_tile_re_label, ntilesX, ntilesY, 1, // Grid
                        // dimension
                        // ::
                        // grid(ntilesX,ntilesY,1);
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        sh_mem, null, // Shared memory size and stream
                        kern_04, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
                if (UrbanGridCUDAProcess.TESTING) {
                    try {
                        Pointer kern_08 = Pointer.to(Pointer.to(lab_mat_gpu),
                                Pointer.to(new int[] { WIDTH_e }),
                                Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu_f),
                                Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }));
                        // Call the kernel function.
                        cuLaunchKernel(F_del_duplicated_lines, ntilesX, ntilesY, 1, // Grid
                                // dimension
                                // ::
                                // grid(ntilesX,ntilesY,1);
                                tiledimX, tiledimY, 1, // Block dimension ::
                                // block(tiledimX,tiledimY,1);
                                0, null, // Shared memory size and stream
                                kern_08, null // Kernel- and extra parameters
                        );
                        cuCtxSynchronize();
                        // end_t = clock();
                        // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                        // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000
                        // ));
                        // elapsed_time += (int)( (double)(end_t - start_t ) /
                        // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                        cuMemcpyDtoH(Pointer.to(lab_mat_cpu_f), lab_mat_gpu_f, sizeUintL_s);
                        UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT,
                                lab_mat_cpu_f, DataBuffer.TYPE_INT, "ssgci__Lrand");

                    } catch (IOException e) {
                        LOGGER.log(Level.WARNING, "Could not save GeoTIFF Sample for testing", e);
                    }
                }

                // ***05***
                // count_labels<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu,bins_gpu);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_05 = Pointer.to(Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu),
                        Pointer.to(bins_gpu));
                // Call the kernel function.
                cuLaunchKernel(F_count_labels, ntilesX, ntilesY, 1, // Grid dimension ::
                        // grid(ntilesX,ntilesY,1);
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        sh_mem, null, // Shared memory size and stream
                        kern_05, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
                cuMemcpyDtoH(Pointer.to(bins_cpu), bins_gpu, sizeBins);
                cumsum[0] = 0;
                int Nbins = bins_cpu[0];
                for (int ii = 1; ii < ntilesX * ntilesY; ii++) {
                    cumsum[ii] = Nbins;
                    Nbins += bins_cpu[ii];
                }
                int Nbins_0 = Nbins + 1;// I count also the ZERO!

                // ***06***
                // labels__1_to_N<<<grid,block,sh_mem>>>( WIDTH_e, HEIGHT_e,
                // lab_mat_gpu, lab_mat_gpu_1N, bins_gpu, Nbins, bdx_e, ID_rand_gpu,
                // ID_1toN_gpu );
                CUdeviceptr ID_rand_gpu = new CUdeviceptr();
                CUdeviceptr ID_1toN_gpu = new CUdeviceptr();
                if (Nbins <= 0) {
                    System.out.println("WARNING: NBins == " + Nbins);
                }
                cuMemAlloc(ID_rand_gpu, Nbins * Sizeof.INT);
                cuMemAlloc(ID_1toN_gpu, Nbins * Sizeof.INT);
                cuMemcpyHtoD(bins_gpu, Pointer.to(cumsum), sizeBins);
                int bdx_e = WIDTH_e - (ntilesX - 1) * tiledimX;
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_06 = Pointer.to(Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu),
                        Pointer.to(lab_mat_gpu_1N), Pointer.to(bins_gpu),
                        Pointer.to(new int[] { Nbins }), Pointer.to(new int[] { bdx_e }),
                        Pointer.to(ID_rand_gpu), Pointer.to(ID_1toN_gpu));
                // Call the kernel function.
                cuLaunchKernel(F_labels__1_to_N, ntilesX, ntilesY, 1, // Grid dimension
                        // ::
                        // grid(ntilesX,ntilesY,1);
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        sh_mem, null, // Shared memory size and stream
                        kern_06, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                if (UrbanGridCUDAProcess.TESTING) {
                    try {
                        int ID_rand_cpu[] = new int[Nbins];
                        int ID_1toN_cpu[] = new int[Nbins];
                        cuMemAllocHost(Pointer.to(ID_rand_cpu), Nbins * Sizeof.INT);
                        cuMemAllocHost(Pointer.to(ID_1toN_cpu), Nbins * Sizeof.INT);
                        cuMemcpyDtoH(Pointer.to(ID_rand_cpu), ID_rand_gpu, Nbins * Sizeof.INT);
                        cuMemcpyDtoH(Pointer.to(ID_1toN_cpu), ID_1toN_gpu, Nbins * Sizeof.INT);
                        storePlainTextSampleFile(ID_rand_cpu, "ssgci_ID_rand_cpu");
                        storePlainTextSampleFile(ID_1toN_cpu, "ssgci_ID_1toN_cpu");
                    } catch (FileNotFoundException e) {
                        LOGGER.log(Level.WARNING, "Could not save Text File Sample for testing", e);
                    }
                }

                // ***07***
                // intratile_relabel_1toN<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu,lab_mat_gpu_1N);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_07 = Pointer.to(Pointer.to(new int[] { WIDTH_e }),
                        Pointer.to(new int[] { HEIGHT_e }), Pointer.to(lab_mat_gpu),
                        Pointer.to(lab_mat_gpu_1N));
                // Call the kernel function.
                cuLaunchKernel(F_intratile_relabel_1toN, ntilesX, ntilesY, 1, // Grid
                        // dimension
                        // ::
                        // grid(ntilesX,ntilesY,1);
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        sh_mem, null, // Shared memory size and stream
                        kern_07, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                // ***08***
                // del_duplicated_lines<<<grid,block>>>(lab_mat_gpu_1N,WIDTH_e,HEIGHT_e,
                // lab_mat_gpu_f,WIDTH,HEIGHT);
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_08 = Pointer.to(Pointer.to(lab_mat_gpu_1N),
                        Pointer.to(new int[] { WIDTH_e }), Pointer.to(new int[] { HEIGHT_e }),
                        Pointer.to(lab_mat_gpu_f), Pointer.to(new int[] { WIDTH }),
                        Pointer.to(new int[] { HEIGHT }));
                // Call the kernel function.
                cuLaunchKernel(F_del_duplicated_lines, ntilesX, ntilesY, 1, // Grid
                        // dimension
                        // ::
                        // grid(ntilesX,ntilesY,1);
                        tiledimX, tiledimY, 1, // Block dimension ::
                        // block(tiledimX,tiledimY,1);
                        0, null, // Shared memory size and stream
                        kern_08, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
                cuMemcpyDtoH(Pointer.to(lab_mat_cpu_f), lab_mat_gpu_f, sizeUintL_s);

                /*
                 * debugging
                 */
                if (UrbanGridCUDAProcess.TESTING) {
                    try {
                        UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT,
                                lab_mat_cpu_f, DataBuffer.TYPE_INT, "ssgci_lab");
                    } catch (IOException e) {
                        LOGGER.log(Level.WARNING, "Could not save GeoTIFF Sample for testing", e);
                    }
                }

                // ***09***
                // histogram_shmem<unsigned int>
                // <<<dimGrid,dimBlock,smemSize0>>>(lab_mat_gpu_f, dev_ROI, d_histogram,
                // map_len, Nbins_0);
                // histogram <unsigned int> <<<dimGrid,dimBlock>>> (lab_mat_gpu_f,
                // dev_ROI, d_histogram, map_len, Nbins_0);
                int h_histogram[] = new int[Nbins_0];
                d_histogram = new CUdeviceptr();
                cuMemAllocHost(Pointer.to(h_histogram), Nbins_0 * Sizeof.INT);
                cuMemAlloc(d_histogram, Nbins_0 * Sizeof.INT);
                cudaMemset(d_histogram, 0, Nbins_0 * Sizeof.INT);

                // code specific for ss-gci:
                cuMemcpyDtoH(Pointer.to(urban_cpu), urban_gpu, sizeChar_o);
                System.arraycopy(urban_cpu, WIDTH, TMP, 0, map_len);// copy urban(which
                // is BINf now) to
                // TMP skipping the
                // first row of
                // zeros.
                cuMemcpyHtoD(d_BINf, Pointer.to(TMP), sizeChar);

                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers which
                // point to the actual values.
                Pointer kern_09 = Pointer.to(Pointer.to(lab_mat_gpu_f), Pointer.to(d_BINf),
                        Pointer.to(d_histogram), Pointer.to(new int[] { map_len }),
                        Pointer.to(new int[] { Nbins_0 }));
                int smemSize0 = (Nbins_0) * Sizeof.INT;
                // Call the kernel function.
                /*
                 * if( smemSize0 < sharedMemPerBlock ){ cuLaunchKernel(F_histogram_shmem, Nblks_per_grid, 1, 1, // Grid dimension :: dimGrid(
                 * Nblks_per_grid, 1, 1 ); threads_ccl, 1, 1, // Block dimension :: dimBlock( threads_ccl, 1, 1 ); smemSize0, null, // Shared memory
                 * size and stream kern_09, null // Kernel- and extra parameters ); }else{
                 */cuLaunchKernel(F_histogram, Nblks_per_grid, 1, 1, // Grid dimension
                        // :: dimGrid(
                        // Nblks_per_grid,
                        // 1, 1 );
                        threads_ccl, 1, 1, // Block dimension :: dimBlock( threads_ccl,
                        // 1, 1 );
                        0, null, // Shared memory size and stream
                        kern_09, null // Kernel- and extra parameters
                );
                // }
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

                cuMemcpyDtoH(Pointer.to(h_histogram), d_histogram, Nbins_0 * Sizeof.INT);

                if (UrbanGridCUDAProcess.TESTING) {
                    try {
                        storePlainTextSampleFile(h_histogram, "ssgci_histogram");
                    } catch (FileNotFoundException e) {
                        LOGGER.log(Level.WARNING, "Could not save Text File Sample for testing", e);
                    }
                }
                // System.out.println("\n CUDA Finished!!\n");
                /*
                 * long estimatedTime = System.currentTimeMillis() - startTime; System.out.println("Elapsed time fragmentation()" + estimatedTime +
                 * " [ms]");
                 */
                /*
                 * h_histogram : histogram counts including the count of zeros (background); Nbins_0 : number of bins, including the count of zeros
                 * (background); lab_mat_cpu_f : map of same size as the one for current admin unit and with labels of the map of current year.
                 */
                // return Arrays.asList(h_histogram, Nbins_0, lab_mat_cpu_f); // ––>
                // lab_mat_cpu_f is useful for testing purpose, maybe not on MapStore!
                Map<String, Object> resultMap = new HashMap<String, Object>();

                resultMap.put("h_histogram", h_histogram);
                resultMap.put("Nbins_0", Nbins_0);
                resultMap.put("lab_mat_cpu_f", lab_mat_cpu_f);

                return resultMap;
            } finally {
                try {

                    // FREE MEMORY:
                    // ...on the host:
                    // It seems that host mem cannot be freed!
                    // cudaFreeHost(lab_mat_cpu);
                    // ...on the device:
                    if (lab_mat_gpu_f != null)
                        cuMemFree(lab_mat_gpu_f);
                    if (lab_mat_gpu != null)
                        cuMemFree(lab_mat_gpu);
                    if (lab_mat_gpu_1N != null)
                        cuMemFree(lab_mat_gpu_1N);
                    if (urban_gpu != null)
                        cuMemFree(urban_gpu);
                    if (dev_ROI != null)
                        cuMemFree(dev_ROI);
                    if (d_histogram != null)
                        cuMemFree(d_histogram);

                    // Unload MODULE
                    if (module != null)
                        cuModuleUnload(module);

                    // Destroy CUDA context:
                    if (context != null)
                        cuCtxDestroy(context);
                } catch (Exception e) {
                    // LOG THE EXCEPTION
                }
            }
        }
    }

    /**
     * @param gpuDeviceCount
     * @return
     * @throws ProcessException
     */
    protected static int getGPUDeviceCount() throws ProcessException {
        int gpuDeviceCount[] = { 0 };
        /*
         * ESTABILISH CONTEXT
         */
        JCudaDriver.setExceptionsEnabled(true);
        // Initialise the driver:
        // err =
        cuInit(0);

        /*
         * RECOGNIZE DEVICE(s) EXISTENCE:
         */
        /*
         * if (err == CUDA_SUCCESS) CUDA_CHECK_RETURN(cuDeviceGetCount(gpuDeviceCount)); if (deviceCount == 0) {
         * System.out.println("Error: no devices supporting CUDA\n"); exit(-1); }
         */
        // Obtain the number of devices
        cuDeviceGetCount(gpuDeviceCount);
        int deviceCount = gpuDeviceCount[0];
        if (deviceCount == 0) {
            throw new ProcessException("Error: no devices supporting CUDA.");
        }

        return deviceCount;
    }

    private static void storeIndex_double(double INDEX, String name) throws FileNotFoundException {
        final File file = new File(UrbanGridCUDAProcess.TESTING_DIR, name/*
                                                                          * +(System . nanoTime ())
                                                                          */
                + ".txt");
        PrintWriter pr = new PrintWriter(file);
        pr.println(INDEX);
        pr.close();
    }

    private static void storePlainTextSampleFile(int[] data, String name)
            throws FileNotFoundException {
        final File file = new File(UrbanGridCUDAProcess.TESTING_DIR, name/*
                                                                          * +(System . nanoTime ())
                                                                          */
                + ".txt");
        PrintWriter pr = new PrintWriter(file);

        for (int i = 0; i < data.length; i++) {
            pr.println(data[i]);
        }
        pr.close();
    }

    private static void storePlainTextSampleFile_double(double[] data, String name)
            throws FileNotFoundException {
        final File file = new File(UrbanGridCUDAProcess.TESTING_DIR, name/*
                                                                          * +(System . nanoTime ())
                                                                          */
                + ".txt");
        PrintWriter pr = new PrintWriter(file);

        for (int i = 0; i < data.length; i++) {
            pr.println(data[i]);
        }
        pr.close();
    }

    /**
     * 
     * @param beans :: list of layers/stuff and their properties
     * @param year :: index defining the year in which calculate perimeter (reference, current)
     * @param admin_unit :: index defining the administrative unit which is undergoing calculation
     * @return :: the overall perimeter of all objects within BIN & ROI
     * @throws IOException
     */
    private static int perimeter(List<CUDABean> beans, int year, int admin_unit) throws IOException {
        /**
         * Copyright 2016 Giuliano Langella ---- Jcuda version works fine!! ----
         * 
         * This function computes the perimeter in driver-API CUDA-C. ************************* -0- bin_x_roi ––> BINf = BIN x ROI -1- gtranspose = f(
         * BIN ––> ROI ) :: "|" –> "––" -2- tidx2_ns = f( ROI ––> TMP ) :: "––" –> "––" {+East/Ovest} -3- gtranspose = f( TMP ––> PERI ) :: "––" –>
         * "|" -4- tidx2_ns = f( BIN ––> PERI ) :: "|" –> "|" {+North/South} -5- reduce6 = f( PERI,ROI ) :: "|*|" –> "∑Perimeter"
         * *************************
         * 
         * TWO IMPORTANT NOTES: (1) I need to calculate BINf = BIN*ROI at the beginning (2) The kernel reduce6 does not work fine!! This means that
         * the map accounting for perimeter (i.e. PERI) is fine (using the logical(.) function in MatLab gives exactly the same result!!).
         */

        /*
         * NOTES The CUDA algorithm manage the object borders at the map borders as they were surrounded by non-object pixels. This mean that a
         * 1-pixel object on the map border has perimeter 4 and not 3. This should be the same approach used by the MatLab "bwperim" built-in
         * function. This choice was taken to solve the current trouble, where the *1 must be accounted properly: | ... | | 0 0 0 ... | | 1 1 1 ... |
         * |*1 1 1 ... | | 1 1 1 ... | | 0 0 0 ... | | ... | The *1 pixel would contribute to the calculation of overall perimeter with "zero" value,
         * while the most accurate for my purpose is a value equal to "one". This can happen only considering that outside the map we assume every
         * pixel is zero, as the following: 0 | ... | 0 | 0 0 0 ... | 0 | 1 1 1 ... | 0 |*1 1 1 ... | 0 | 1 1 1 ... | 0 | 0 0 0 ... | 0 | ... | This
         * pattern is valid for both East–Ovest and North–South searching directions.
         * 
         * The CUDA algorithm is based on running tidx2_ns kernel twice for: (1) East–West dir (after transpose "|" ––> "––") (2) North–South dir
         * (after transpose "––" ––> "|") The tidx2_ns kernel performs the algebraic sum of three rows: > the pixel at top of current pixel + IN[
         * tid+(ii-1)+map_width ] > the pixel at bottom of current pixel - IN[ tid+(ii+1)+map_width ] > the current pixel +2*IN[ tid+(ii+0)*map_width
         * ]
         * 
         * This is the cheapest chain of calculations (in terms of gpu storage) that sets the required arrays: #1 #2 #3 #4 #5 BIN ––(transpose)––> ROI
         * ––––(2TID-NS)––> TMP ––(transpose)––> PERI || BIN ––(2TID-NS)––> PERI & ROI ––(reduce)––> ∑Perimeter char char double doub char doub & char
         * double |––>load ROI now!!
         */
        /*
         * PARAMETERS
         */
        CUresult err;
        CUcontext context = null;
        CUmodule module = null;
        CUdeviceptr dev_PERI = null;
        CUdeviceptr dev_BIN = null;
        CUdeviceptr dev_BINf = null;
        CUdeviceptr dev_ROI = null;
        CUdeviceptr dev_TMP = null;
        CUdeviceptr d_Perimeter = null;

        // int gpuDeviceCount[] = { 0 };
        int elapsed_time = 0;
        // count the number of kernels that must print their output:
        int count_print = 0;
        // System.out.println("Remember that you have to activate the division by std_4_area in the cuda kernel (mask_twice)!!\n");
        // rural = true; System.out.println("delete 'rural = true' from code.");

        try {
            int deviceCount = getGPUDeviceCount();

            /*
             * SELECT THE FIRST DEVICE (but I should select/split devices according to streams)
             */
            int selDev = getRandomGpuDevice(deviceCount);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, selDev);
            // Query some useful properties:
            int amountProperty[] = { 0 };
            // -1-
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            int maxThreadsPerBlock = amountProperty[0];
            // -2-
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
            int MaxGridDimX = amountProperty[0];
            // -3-
            // ...others?

            /*
             * CREATE THE CONTEXT (for currently selected device)
             */
            context = new CUcontext();
            // int cuCtxCreate_STATUS =
            cuCtxCreate(context, selDev, device);
            // Load the ptx file:
            module = new CUmodule();
            cuModuleLoad(module, PTXFILE_perimeter);
            /*
             * ENTRY POINTS (obtain function pointers to the entry kernels) _Z15gtranspose_charPhPKhjj, _Z17gtranspose_doublePdPKdjj,
             * _Z15tidx2_ns_doublePKhjjPdj, _Z14reduce6_doublePKdPKhPdjjb
             */
            // BIN x ROI
            CUfunction F_bin_x_roi = new CUfunction();
            cuModuleGetFunction(F_bin_x_roi, module, "_Z9bin_x_roiPKhS0_Phjj");
            // gtranspose_char
            CUfunction F_gtranspose_char = new CUfunction();
            cuModuleGetFunction(F_gtranspose_char, module, "_Z15gtranspose_charPhPKhjj");
            // gtranspose_double
            CUfunction F_gtranspose_double = new CUfunction();
            cuModuleGetFunction(F_gtranspose_double, module, "_Z17gtranspose_doublePdPKdjj");
            // tidx2_ns_double
            CUfunction F_tidx2_ns_double = new CUfunction();
            cuModuleGetFunction(F_tidx2_ns_double, module, "_Z15tidx2_ns_doublePKhjjPdj");
            // reduce6_double
            CUfunction F_reduce6_double = new CUfunction();
            cuModuleGetFunction(F_reduce6_double, module, "_Z14reduce6_doublePKdPdjjb");

            /*
             * DIM of ARRAYS in BYTES
             */
            int WIDTH = beans.get(admin_unit).width;
            int HEIGHT = beans.get(admin_unit).height;
            int map_len = WIDTH * HEIGHT;
            long sizeChar = map_len * Sizeof.BYTE;
            long sizeDouble = map_len * Sizeof.DOUBLE;

            /*
             * CPU ARRAYS
             */
            double host_PERI[] = new double[map_len];
            double h_print_double[] = new double[map_len];
            byte h_print_uchar[] = new byte[map_len];
            double h_Perimeter[] = new double[blocks_pi];
            cuMemAllocHost(Pointer.to(host_PERI), sizeDouble);
            cuMemAllocHost(Pointer.to(h_print_double), sizeDouble);
            cuMemAllocHost(Pointer.to(h_print_uchar), sizeChar);
            cuMemAllocHost(Pointer.to(h_Perimeter), blocks_pi * Sizeof.DOUBLE);
            /*
             * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
             */
            // get pointers
            dev_PERI = new CUdeviceptr();
            dev_BIN = new CUdeviceptr();
            dev_BINf = new CUdeviceptr();
            dev_ROI = new CUdeviceptr();
            dev_TMP = new CUdeviceptr();
            d_Perimeter = new CUdeviceptr();
            // allocate in mem
            cuMemAlloc(dev_PERI, sizeDouble);
            cuMemAlloc(dev_BIN, sizeChar);
            cuMemAlloc(dev_BINf, sizeChar);
            cuMemAlloc(dev_ROI, sizeChar);
            cuMemAlloc(dev_TMP, sizeDouble);
            cuMemAlloc(d_Perimeter, blocks_pi * Sizeof.DOUBLE);
            // set mem
            cudaMemset(dev_TMP, 0, sizeDouble);
            /*
             * MEM COPY H2D
             */
            // double[][] results = new double[year][map_len];
            if (year == 0) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getReferenceImage()),
                        sizeChar);
            } else if (year == 1) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getCurrentImage()), sizeChar);
            } else if (year == 2) {
                // do nothing because the DIFF is calculated in JAVA ==> make the
                // JCuda/Cuda version to speedup computation!!

                /*
                 * System.out.println( "You should implement in CUDA the difference between ref & curr!!" ); for(int ii=0;ii<map_len;ii++){
                 * host_PERI[ii] = beans.get(admin_unit ).referenceImage[ii]-beans.get(admin_unit).getCurrentImage()[ii]; } return host_PERI;
                 */
            }
            cuMemcpyHtoD(dev_ROI, Pointer.to(beans.get(admin_unit).roi), sizeChar);

            /*
             * KERNELS GEOMETRY NOTE: use ceil() instead of the "%" operator!!!
             */
            int BDX = BLOCKDIM_X; // sqrt_nmax_threads = (int) Math.floor(Math.sqrt(
                                  // maxThreadsPerBlock ));
            int gdx_kTidx2NS, gdy_kTidx2NS, gdx_trans, gdy_trans, gdx_kTidx2NS_t, gdy_kTidx2NS_t;
            // k(gtransform)
            gdx_trans = (((WIDTH % BDX) > 0) ? 1 : 0) + WIDTH / BDX;
            gdy_trans = (((HEIGHT % BDX) > 0) ? 1 : 0) + HEIGHT / BDX;
            // k(2*TID - NS)
            gdx_kTidx2NS = (((WIDTH % (BDX * BDX)) > 0) ? 1 : 0) + (WIDTH / (BDX * BDX));
            gdy_kTidx2NS = (((HEIGHT % mask_len_pi) > 0) ? 1 : 0)
                    + (int) Math.floor(HEIGHT / mask_len_pi);
            gdx_kTidx2NS_t = (((HEIGHT % (BDX * BDX)) > 0) ? 1 : 0) + (HEIGHT / (BDX * BDX));
            gdy_kTidx2NS_t = (((WIDTH % mask_len_pi) > 0) ? 1 : 0)
                    + (int) Math.floor(WIDTH / mask_len_pi);
            // k(reduce6)
            int smemSize = (threads_pi <= 32) ? (2 * threads_pi * Sizeof.DOUBLE)
                    : (threads_pi * Sizeof.DOUBLE);

            /*
             * KERNELS INVOCATION
             * 
             * ************************* -1- gtranspose = ƒ( BIN ––> ROI ) :: "|" –> "––" -2- tidx2_ns = ƒ( ROI ––> TMP ) :: "––" –> "––"
             * {+East/Ovest} -3- gtranspose = ƒ( TMP ––> PERI ) :: "––" –> "|" -4- tidx2_ns = ƒ( BIN ––> PERI ) :: "|" –> "|" {+North/South} -5-
             * reduce6 = ƒ( PERI,ROI ) :: "|*|" –> "∑Perimeter" ************************* #1 #2 #3 #4 #5 BIN ––(transpose)––> ROI ––––(2TID-NS)––> TMP
             * ––(transpose)––> PERI || BIN ––(2TID-NS)––> PERI & ROI ––(reduce)––> ∑Perimeter
             */

            // ***00***
            // bin_x_roi<<< dimGrid, dimBlock >>>( dev_BIN, dev_ROI, dev_BINf,
            // map_len, 512 );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_bin_x_roi = Pointer.to(Pointer.to(dev_BIN), Pointer.to(dev_ROI),
                    Pointer.to(dev_BINf), Pointer.to(new int[] { map_len }),
                    Pointer.to(new int[] { threads_pi }));
            // Call the kernel function.
            cuLaunchKernel(F_bin_x_roi, blocks_pi, 1, 1, // Grid dimension :: dim3
                                                         // dimGrid( blocks, 1, 1
                                                         // );
                    threads_pi, 1, 1, // Block dimension :: dim3 dimBlock( threads,
                                      // 1, 1 );
                    0, null, // Shared memory size and stream
                    kern_bin_x_roi, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();

            // ***01***
            // start_t = clock();
            // gtranspose_char<<<grid_trans,block_trans>>>( dev_ROI, dev_BIN,
            // MDbin.width, MDbin.heigth );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_gtrans = Pointer.to(Pointer.to(dev_ROI), Pointer.to(dev_BINf),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }));
            // Call the kernel function.
            cuLaunchKernel(F_gtranspose_char, gdx_trans, gdy_trans, 1,// Grid
                                                                      // dimension
                                                                      // :: dim3
                                                                      // grid_trans
                                                                      // (gdx_trans,gdy_trans);
                    BDX, BDX, 1, // Block dimension :: dim3 block_trans( BDX,
                                 // BDX,1);
                    0, null, // Shared memory size and stream
                    kern_gtrans, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            // end_t = clock();
            // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
            // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
            // elapsed_time += (int)( (double)(end_t - start_t ) /
            // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

            // ***02***
            // tidx2_ns<double><<<grid_kTidx2NS_t,block_kTidx2NS_t>>>( dev_ROI,
            // MDbin.heigth, MDbin.width, dev_TMP, mask_len );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_tidx2ns = Pointer.to(Pointer.to(dev_ROI),
                    Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH }),
                    Pointer.to(dev_TMP), Pointer.to(new int[] { mask_len_pi }));
            // Call the kernel function.
            cuLaunchKernel(F_tidx2_ns_double, gdx_kTidx2NS_t, gdy_kTidx2NS_t, 1, // Grid
                                                                                 // dimension
                                                                                 // ::
                                                                                 // dim3
                                                                                 // grid_kTidx2NS_t
                                                                                 // (gdx_kTidx2NS_t,gdy_kTidx2NS_t,1);
                    BDX * BDX, 1, 1, // Block dimension :: dim3
                                     // block_kTidx2NS_t(BDX*BDX,1,1);
                    0, null, // Shared memory size and stream
                    kern_tidx2ns, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_13,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            // ***03***
            // gtranspose_double<<<grid_trans2,block_trans>>>( dev_PERI, dev_TMP,
            // MDbin.heigth, MDbin.width );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_gtrans_t = Pointer.to(Pointer.to(dev_PERI), Pointer.to(dev_TMP),
                    Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH }));
            // Call the kernel function.
            cuLaunchKernel(F_gtranspose_double, gdy_trans, gdx_trans, 1, // Grid
                                                                         // dimension
                                                                         // ::
                                                                         // dim3
                                                                         // grid_trans2(gdy_trans,gdx_trans);
                    BDX, BDX, 1, // Block dimension :: dim3 block_trans( BDX, BDX,1
                                 // );
                    0, null, // Shared memory size and stream
                    kern_gtrans_t, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_trans,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            // ***04***
            // tidx2_ns<double><<<grid_kTidx2NS,block_kTidx2NS>>>( dev_BIN,
            // MDbin.width, MDbin.heigth, dev_PERI, mask_len );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_tidx2ns_t = Pointer.to(Pointer.to(dev_BINf),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(dev_PERI), Pointer.to(new int[] { mask_len_pi }));
            // Call the kernel function.
            cuLaunchKernel(F_tidx2_ns_double, gdx_kTidx2NS, gdy_kTidx2NS, 1, // Grid
                                                                             // dimension
                                                                             // ::
                                                                             // grid_kTidx2NS
                                                                             // (gdx_kTidx2NS,gdy_kTidx2NS,1);
                    BDX * BDX, 1, 1, // Block dimension :: dim3 block_kTidx2NS(
                                     // BDX*BDX,1,1 );
                    0, null, // Shared memory size and stream
                    kern_tidx2ns_t, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_13,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            // ***05***
            // reduce6_double<<< dimGrid, dimBlock, smemSize >>>( dev_PERI,
            // d_Perimeter, map_len, 512, isPow2(map_len) );
            // reduce6_double<<<dimGrid, dimBlock, smemSize>>>( dev_PERI, dev_ROI,
            // d_Perimeter, map_len, threads_pi, nIsPow2 )
            // start_t = clock();
            byte nIsPow2 = (byte) (((map_len & (map_len - 1)) == 0) ? 1 : 0);
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_reduce6 = Pointer.to(Pointer.to(dev_PERI), Pointer.to(d_Perimeter),
                    Pointer.to(new int[] { map_len }), Pointer.to(new int[] { threads_pi }),
                    Pointer.to(new byte[] { nIsPow2 }));
            // Call the kernel function.
            cuLaunchKernel(F_reduce6_double, blocks_pi, 1, 1, // Grid dimension ::
                                                              // dimGrid(blocks,
                                                              // 1, 1 );
                    threads_pi, 1, 1, // Block dimension :: dim3 dimBlock(
                                      // threads,1, 1 );
                    smemSize, null, // Shared memory size and stream
                    kern_reduce6, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_24,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            /*
             * System.out.println("______________________________________\n"); System.out.println("  %21s\t%6d [msec]\n",
             * "Total time (T):",elapsed_time );
             */

            // extract perimeter:
            int sumPerimeter = 0;
            cuMemcpyDtoH(Pointer.to(h_Perimeter), d_Perimeter, blocks_pi * Sizeof.DOUBLE);
            for (int ii = 0; ii < blocks_pi; ii++) {
                sumPerimeter += h_Perimeter[ii];
            }
            // System.out.println("Perimeter = %d\n\n",sumPerimeter);

            cuMemcpyDtoH(Pointer.to(host_PERI), dev_PERI, sizeDouble);
            if (UrbanGridCUDAProcess.TESTING) {
                UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT, host_PERI,
                        DataBuffer.TYPE_DOUBLE, "ssgci__PERI");
            }
            // System.out.println("\n CUDA Finished!!\n");
            /*
             * long estimatedTime = System.currentTimeMillis() - startTime; System.out.println("Elapsed time fragmentation()" + estimatedTime +
             * " [ms]");
             */
            return sumPerimeter; // --> return host_PERI

        } finally {
            try {
                // CUDA free:
                if (dev_BIN != null)
                    cuMemFree(dev_BIN);
                if (dev_BINf != null)
                    cuMemFree(dev_BINf);
                if (dev_ROI != null)
                    cuMemFree(dev_ROI);
                if (dev_PERI != null)
                    cuMemFree(dev_PERI);
                if (dev_TMP != null)
                    cuMemFree(dev_TMP);
                if (d_Perimeter != null)
                    cuMemFree(d_Perimeter);

                // Unload MODULE
                if (module != null)
                    cuModuleUnload(module);

                // Destroy CUDA context:
                if (context != null)
                    cuCtxDestroy(context);
            } catch (Exception e) {
                // log thr exception
            }
        }
    }

    /**
     * This function calculates the rural/urbal fragmentation according to the user-defined radius.
     * 
     * @param beans :: list of layers and their properties
     * @param rural :: flag for rural(=TRUE)/urban(=FALSE) switch
     * @param areaOfOnePixel :: pixel area in m2
     * @param RADIUS :: radius of counting as number of pixels
     * @param year :: index defining the year in which calculate fragmentation (reference, current)
     * @param admin_unit :: index defining the administrative unit which is undergoing calculation
     * @return
     */
    public static double[] fragmentation(List<CUDABean> beans, boolean rural, int RADIUS, int year,
            int admin_unit) {
        /**
         * Copyright 2015 Giuliano Langella: ---- working fine!! ----
         * 
         * This function computes the fragmentation in driver-API CUDA-C. -1- complementary_to_ONE { BIN,ONE } --> COMP -2- gtranspose { BIN } -3-
         * Vcumsum { BIN } --> FRAG -4- sum_of_3_LINES { FRAG } -5- gtranspose { FRAG } -6- Vcumsum { FRAG } -7- sum_of_3_LINES { FRAG } -8-
         * mask_twice { FRAG,COMP,ROI } --> FRAG
         */

        /*
         * NOTES I should use a more performing transpose algorithm (maybe based on shared mem). The best way to deal with more administrative units
         * is to use cuda streams in parallel. At now, for simplicity I assume that the fragmentation JCuda class is called for any admin unit from
         * outside. But to allow an intermediate implementation, I pass all the admin units to the function, and I extract the useful info for the
         * first admin unit. After I have to enlarge the code for using streams.
         */

        // long startTime = System.currentTimeMillis();

        /*
         * PARAMETERS
         */
        CUresult err;
        CUcontext context = null;
        CUmodule module = null;
        CUdeviceptr dev_FRAG = null;
        CUdeviceptr dev_IO = null;
        CUdeviceptr dev_BIN = null;
        CUdeviceptr dev_ROI = null;
        CUdeviceptr dev_TMP = null;
        CUdeviceptr dev_COMP = null;
        CUdeviceptr dev_ONE = null;

        int mask_len = RADIUS * 2 + 1;
        double std_4_area = mask_len * mask_len; // (RADIUS*RADIUS*areaOfOnePixel);
        // int gpuDeviceCount[] = { 0 };
        int elapsed_time = 0;
        // count the number of kernels that must print their output:
        int count_print = 0;
        // System.out.println("Remember that you have to activate the division by std_4_area in the cuda kernel (mask_twice)!!\n");
        // rural = true; System.out.println("delete 'rural = true' from code.");

        try {
            int deviceCount = getGPUDeviceCount();

            /*
             * SELECT THE FIRST DEVICE (but I should select/split devices according to streams)
             */
            int selDev = getRandomGpuDevice(deviceCount);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, selDev);
            // Query some useful properties:
            int amountProperty[] = { 0 };
            // -1-
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            int maxThreadsPerBlock = amountProperty[0];
            // -2-
            // ...others?

            /*
             * CREATE THE CONTEXT (for currently selected device)
             */
            context = new CUcontext();
            // int cuCtxCreate_STATUS =
            cuCtxCreate(context, selDev, device);
            // Load the ptx file:
            module = new CUmodule();
            cuModuleLoad(module, PTXFILE_fragmentation);
            /*
             * ENTRY POINTS (obtain a function pointer to the -5- entry functions/kernels)
             */
            // complementary_to_ONE
            CUfunction F_complementary_to_ONE = new CUfunction();
            cuModuleGetFunction(F_complementary_to_ONE, module,
                    "_Z20complementary_to_ONEPKhS0_Phjj");
            // gtranspose_char
            CUfunction F_gtranspose_char = new CUfunction();
            cuModuleGetFunction(F_gtranspose_char, module, "_Z15gtranspose_charPhPKhjj");
            // gtranspose_double
            CUfunction F_gtranspose_double = new CUfunction();
            cuModuleGetFunction(F_gtranspose_double, module, "_Z17gtranspose_doublePdPKdjj");
            // Vcumsum_char
            CUfunction F_Vcumsum_char = new CUfunction();
            cuModuleGetFunction(F_Vcumsum_char, module, "_Z12Vcumsum_charPKhmmPdj");
            // Vcumsum_double
            CUfunction F_Vcumsum_double = new CUfunction();
            cuModuleGetFunction(F_Vcumsum_double, module, "_Z14Vcumsum_doublePKdmmPdj");
            // sum_of_3_LINES
            CUfunction F_sum_of_3_LINES = new CUfunction();
            cuModuleGetFunction(F_sum_of_3_LINES, module, "_Z14sum_of_3_LINESPKdjjPdj");
            // mask_twice
            CUfunction F_mask_twice = new CUfunction();
            cuModuleGetFunction(F_mask_twice, module, "_Z10mask_twicePdPKhS1_jjd");

            /*
             * DIM of ARRAYS in BYTES
             */
            int WIDTH = beans.get(admin_unit).width;
            int HEIGHT = beans.get(admin_unit).height;
            int map_len = WIDTH * HEIGHT;
            long sizeChar = map_len * Sizeof.BYTE;
            long sizeDouble = map_len * Sizeof.DOUBLE;
            /*
             * CPU ARRAYS
             */
            double host_FRAG[] = new double[map_len];
            double host_IO[] = new double[map_len];
            byte host_TMP[] = new byte[map_len];
            byte host_COMP[] = new byte[map_len];
            cuMemAllocHost(Pointer.to(host_FRAG), sizeDouble);
            cuMemAllocHost(Pointer.to(host_IO), sizeDouble);
            cuMemAllocHost(Pointer.to(host_TMP), sizeChar);
            cuMemAllocHost(Pointer.to(host_COMP), sizeChar);
            /*
             * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
             */
            // get pointers
            dev_FRAG = new CUdeviceptr();
            dev_IO = new CUdeviceptr();
            dev_BIN = new CUdeviceptr();
            dev_ROI = new CUdeviceptr();
            dev_TMP = new CUdeviceptr();
            dev_COMP = new CUdeviceptr();
            dev_ONE = new CUdeviceptr();
            // allocate in mem
            cuMemAlloc(dev_FRAG, sizeDouble);
            cuMemAlloc(dev_IO, sizeDouble);
            cuMemAlloc(dev_BIN, sizeChar);
            cuMemAlloc(dev_ROI, sizeChar);
            cuMemAlloc(dev_TMP, sizeChar);
            cuMemAlloc(dev_COMP, sizeChar);
            cuMemAlloc(dev_ONE, sizeChar);
            // set mem
            cudaMemset(dev_FRAG, 0, sizeDouble);
            cudaMemset(dev_IO, 0, sizeDouble);
            cudaMemset(dev_COMP, 0, sizeChar);
            cudaMemset(dev_ONE, 1, sizeChar);
            /*
             * MEM COPY H2D
             */
            // double[][] results = new double[year][map_len];
            if (year == 0) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getReferenceImage()),
                        sizeChar);
            } else if (year == 1) {
                cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getCurrentImage()), sizeChar);
            } else if (year == 2) {
                // do nothing because the DIFF is calculated in JAVA

                /*
                 * System.out.println( "You should implement in CUDA the difference between ref & curr!!" ); for(int ii=0;ii<map_len;ii++){
                 * host_FRAG[ii] = beans.get(admin_unit ).referenceImage[ii]-beans.get(admin_unit).getCurrentImage()[ii]; } return host_FRAG;
                 */
            }
            cuMemcpyHtoD(dev_ROI, Pointer.to(beans.get(admin_unit).roi), sizeChar);

            /*
             * KERNELS GEOMETRY NOTE: use ceil() instead of the "%" operator!!!
             */
            int BDX = BLOCKDIM_X; //
            int gdx_k12, gdy_k12, gdx_k3, gdy_k3, gdx_trans, gdy_trans, gdx_k12_t, gdy_k12_t, gdx_mask, gdy_mask;
            // k1 + k2
            gdx_k12 = (((WIDTH % mask_len) > 0) ? 1 : 0) + (WIDTH / mask_len);
            gdy_k12 = (((HEIGHT % (BDX * BDX)) > 0) ? 1 : 0)
                    + (int) Math.floor(HEIGHT / (BDX * BDX));
            // k3 + k4
            gdx_k3 = (((WIDTH % (BDX * BDX)) > 0) ? 1 : 0) + (WIDTH / (BDX * BDX));
            gdy_k3 = (((HEIGHT % mask_len) > 0) ? 1 : 0) + (int) Math.floor(HEIGHT / mask_len);
            gdx_k12_t = (((HEIGHT % (BDX * BDX)) > 0) ? 1 : 0) + (HEIGHT / (BDX * BDX));
            gdy_k12_t = (((WIDTH % mask_len) > 0) ? 1 : 0) + (int) Math.floor(WIDTH / mask_len);
            // k(gtransform)
            gdx_trans = (((WIDTH % BDX) > 0) ? 1 : 0) + WIDTH / BDX;
            gdy_trans = (((HEIGHT % BDX) > 0) ? 1 : 0) + HEIGHT / BDX;
            // mask_twice
            gdx_mask = (((WIDTH % BDX) > 0) ? 1 : 0) + WIDTH / BDX;
            gdy_mask = (((HEIGHT % BDX) > 0) ? 1 : 0) + HEIGHT / BDX;

            /*
             * ALTERNATIVE ALGORITHM TRY using matrix transpose to use the cumsum_vertical & sum_of_3_rows twice: once for step 3 & 4 as regularly
             * done, and the other one for step 1 & 2 in place of kernels that are too slow (cumsum_horizontal & sum_of_3_cols).
             * 
             * Speed tests demonstrate that working along Y when doing cumulative sum is 10 times more efficient: this is true because warps of
             * threads R/W in coalesced patterns.
             * 
             * Try to apply ROI at the end, so that I skip one gtranspose at the beginning.
             */
            // complementary_to_ONE<unsigned char><<<grid_compl,block_compl>>>(
            // dev_ONE,dev_BIN, dev_COMP, MDbin.width, MDbin.heigth );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_compl = Pointer.to(Pointer.to(dev_ONE), Pointer.to(dev_BIN),
                    Pointer.to(dev_COMP), Pointer.to(new int[] { WIDTH }),
                    Pointer.to(new int[] { HEIGHT }));

            // Call the kernel function.
            cuLaunchKernel(F_complementary_to_ONE, gdx_mask, gdy_mask, 1, // Grid
                                                                          // dimension
                    BDX, BDX, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_compl, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            // end_t = clock();
            // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
            // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
            // elapsed_time += (int)( (double)(end_t - start_t ) /
            // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

            if (rural == true) {
                /**
                 * This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels): FRAG = fragmentation_prog(
                 * BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES" FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
                 * * This means that the use of BIN & COMP is straightforward:
                 */
                // gtranspose_char<<<grid_trans,block_trans>>>( dev_TMP2, dev_BIN,
                // MDbin.width, MDbin.heigth );
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers
                // which point to the actual values.
                Pointer kern_gtrans = Pointer.to(Pointer.to(dev_TMP), Pointer.to(dev_BIN),
                        Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }));
                // Call the kernel function.
                cuLaunchKernel(F_gtranspose_char, gdx_trans, gdy_trans, 1,// Grid
                                                                          // dimension
                        BDX, BDX, 1, // Block dimension
                        0, null, // Shared memory size and stream
                        kern_gtrans, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                // end_t = clock();
                // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)(
                // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
                // elapsed_time += (int)( (double)(end_t - start_t ) /
                // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
            } else if (rural == false) {
                /**
                 * This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels): FRAG = fragmentation_prog(
                 * COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES" FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
                 * This means that I have to invert BIN & COMP:
                 */
                // gtranspose_char<<<grid_trans,block_trans>>>( dev_TMP2, dev_BIN,
                // MDbin.width, MDbin.heigth );
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers
                // which point to the actual values.
                Pointer kern_gtrans = Pointer.to(Pointer.to(dev_TMP), Pointer.to(dev_COMP),
                        Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }));
                // Call the kernel function.
                cuLaunchKernel(F_gtranspose_char, gdx_trans, gdy_trans, 1,// Grid
                                                                          // dimension
                        BDX, BDX, 1, // Block dimension
                        0, null, // Shared memory size and stream
                        kern_gtrans, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                /*
                 * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++ count_print,kern_trans_,(int)( (double)(end_t - start_t ) /
                 * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed
                 * time [ms]:
                 */
            }

            // Vcumsum_char<<<grid_k12_t,block_k12_t>>>( dev_TMP2,
            // MDbin.heigth,MDbin.width,dev_FRAG,RADIUS );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_vsum = Pointer.to(Pointer.to(dev_TMP), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(dev_FRAG),
                    Pointer.to(new int[] { RADIUS }));
            // Call the kernel function.
            cuLaunchKernel(F_Vcumsum_char, gdx_k12_t, gdy_k12_t, 1,// Grid dimension
                    BDX * BDX, 1, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_vsum, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_13,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */
            // sum_of_3_LINES<<<grid_k12_t,block_k12_t>>>( dev_FRAG, MDbin.heigth,
            // MDbin.width, dev_IO, RADIUS );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_s3L = Pointer.to(Pointer.to(dev_FRAG), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(dev_IO),
                    Pointer.to(new int[] { RADIUS }));
            // Call the kernel function.
            cuLaunchKernel(F_sum_of_3_LINES, gdx_k12_t, gdy_k12_t, 1,// Grid
                                                                     // dimension
                    BDX * BDX, 1, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_s3L, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            // sum_of_3_LINES<<<grid_k12_t,block_k12_t>>>( dev_FRAG, MDbin.heigth,
            // MDbin.width, dev_IO, RADIUS );
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_24,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */
            // gtranspose_double<<<grid_trans2,block_trans>>>(dev_FRAG, dev_IO,
            // MDbin.heigth, MDbin.width);
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_gtrans = Pointer.to(Pointer.to(dev_FRAG), Pointer.to(dev_IO),
                    Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH }));
            // Call the kernel function.
            cuLaunchKernel(F_gtranspose_double, gdy_trans, gdx_trans, 1,// Grid
                                                                        // dimension
                    BDX, BDX, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_gtrans, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_trans,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */
            // Vcumsum_double<<<grid_k3,block_k3>>>( dev_FRAG,
            // MDbin.width,MDbin.heigth,dev_IO,RADIUS ); // { ",unsigned char" ;
            // "dev_ROI, " }
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_vsum2 = Pointer.to(Pointer.to(dev_FRAG), Pointer.to(new int[] { WIDTH }),
                    Pointer.to(new int[] { HEIGHT }), Pointer.to(dev_IO),
                    Pointer.to(new int[] { RADIUS }));
            // Call the kernel function.
            cuLaunchKernel(F_Vcumsum_double, gdx_k3, gdy_k3, 1, // Grid dimension
                    BDX * BDX, 1, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_vsum2, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_13,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */
            // sum_of_3_LINES<<<grid_k3,block_k3>>>( dev_IO, MDbin.width,
            // MDbin.heigth, dev_FRAG, RADIUS );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_s3L2 = Pointer.to(Pointer.to(dev_IO), Pointer.to(new int[] { WIDTH }),
                    Pointer.to(new int[] { HEIGHT }), Pointer.to(dev_FRAG),
                    Pointer.to(new int[] { RADIUS }));
            // Call the kernel function.
            cuLaunchKernel(F_sum_of_3_LINES, gdx_k3, gdy_k3, 1, // Grid dimension
                    BDX * BDX, 1, 1, // Block dimension
                    0, null, // Shared memory size and stream
                    kern_s3L2, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_24,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */
            if (rural == true) {
                /**
                 * This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels): FRAG = fragmentation_prog(
                 * BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES" FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
                 * * This means that the use of BIN & COMP is straightforward:
                 */
                // mask_twice<<<grid_mask,block_mask>>>( dev_FRAG, dev_ROI,
                // dev_COMP, MDbin.width, MDbin.heigth, std_4_area );
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers
                // which point to the actual values.
                Pointer kern_m2 = Pointer.to(Pointer.to(dev_FRAG), Pointer.to(dev_ROI),
                        Pointer.to(dev_COMP), Pointer.to(new int[] { WIDTH }),
                        Pointer.to(new int[] { HEIGHT }), Pointer.to(new double[] { std_4_area }));
                // Call the kernel function.
                cuLaunchKernel(F_mask_twice, gdx_mask, gdy_mask, 1, // Grid
                                                                    // dimension
                        BDX, BDX, 1, // Block dimension
                        0, null, // Shared memory size and stream
                        kern_m2, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                /*
                 * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++ count_print,kern_mask,(int)( (double)(end_t - start_t ) /
                 * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed
                 * time [ms]:
                 */} else { // if(rural==false){
                /**
                 * This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels): FRAG = fragmentation_prog(
                 * COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES" FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
                 * This means that I have to invert BIN & COMP:
                 */
                // mask_twice<<<grid_mask,block_mask>>>( dev_FRAG, dev_ROI, dev_BIN,
                // MDbin.width, MDbin.heigth, std_4_area );
                // start_t = clock();
                // Set up the kernel parameters: A pointer to an array of pointers
                // which point to the actual values.
                Pointer kern_m2 = Pointer.to(Pointer.to(dev_FRAG), Pointer.to(dev_ROI),
                        Pointer.to(dev_BIN), Pointer.to(new int[] { WIDTH }),
                        Pointer.to(new int[] { HEIGHT }), Pointer.to(new double[] { std_4_area }));
                // Call the kernel function.
                cuLaunchKernel(F_mask_twice, gdx_mask, gdy_mask, 1, // Grid
                                                                    // dimension
                        BDX, BDX, 1, // Block dimension
                        0, null, // Shared memory size and stream
                        kern_m2, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();
                /*
                 * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++ count_print,kern_mask,(int)( (double)(end_t - start_t ) /
                 * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed
                 * time [ms]:
                 */
            }

            /*
             * System.out.println("______________________________________\n"); System.out.println("  %21s\t%6d [msec]\n",
             * "Total time (T):",elapsed_time );
             */

            // ...to be completed:
            cuMemcpyDtoH(Pointer.to(host_FRAG), dev_FRAG, sizeDouble);

            if (UrbanGridCUDAProcess.TESTING) {
                try {
                    double isRural = rural == false ? 0 : 1;
                    UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT, host_FRAG,
                            DataBuffer.TYPE_DOUBLE, "ssgci_FRAG");
                    storeIndex_double(RADIUS * beans.get(0).getCellSize(), "ssgci_radius");
                    storeIndex_double(isRural, "ssgci_rural");
                } catch (IOException e) {
                    LOGGER.log(Level.WARNING,
                            "Could not save Index[10] 'Loss of Food Supply' for testing", e);
                }
            }
            // System.out.println("\n CUDA Finished!!\n");
            /*
             * long estimatedTime = System.currentTimeMillis() - startTime; System.out.println("Elapsed time fragmentation()" + estimatedTime +
             * " [ms]");
             */
            return host_FRAG; // --> return host_FRAG

        } finally {
            try {
                // CUDA free:
                if (dev_FRAG != null)
                    cuMemFree(dev_FRAG);
                if (dev_IO != null)
                    cuMemFree(dev_IO);
                if (dev_BIN != null)
                    cuMemFree(dev_BIN);
                if (dev_ROI != null)
                    cuMemFree(dev_ROI);
                if (dev_TMP != null)
                    cuMemFree(dev_TMP);
                if (dev_COMP != null)
                    cuMemFree(dev_COMP);
                if (dev_ONE != null)
                    cuMemFree(dev_ONE);

                // Unload MODULE
                if (module != null)
                    cuModuleUnload(module);

                // Destroy CUDA context:
                if (context != null)
                    cuCtxDestroy(context);
            } catch (Exception e) {
                // log the exception
            }
        }
    }

    /**
     * 
     * @param beans :: list of layers and their properties
     * @param areaOfOnePixel :: pixel area
     * @param admin_unit :: administrative unit being processed
     * @return a list of two double objects{map of BIN diffs, area [ha] of diffs by 4 types of diffs }
     */
    public static List<double[]> land_take(List<CUDABean> beans, int admin_unit) {
        /**
         * Copyright 2015 Giuliano Langella: ---- testing... ----
         * 
         * This function computes the reduction kernel in driver-API CUDA-C. -1- reduction { BIN1,BIN2,ROI } --> LTAKE_count
         * [kernel::imperviousness_change_histc_sh_4] -2- difference { BIN1,BIN2,ROI } --> LTAKE_map [kernel::imperviousness_change]
         * 
         * ----------------------------- (N) (BIN2,BIN1) --> (LTAKE) ----------------------------- (1) (0,0) --> +0 ---> nothing changed in rural
         * pixels LTAKE_count[0] (2) (0,1) --> -1 ---> increase of rural pixels LTAKE_count[1] (3) (1,0) --> +1 ---> increase of urban pixels
         * LTAKE_count[2] (4) (1,1) --> -0 ---> nothing changed in urban pixels LTAKE_count[3] ----------------------------- where values can be {
         * 0:rural; 1:urban }.
         */

        /*
         * NOTES 1.\ The imperviousness_change_large kernel does not work! So I have to limit the number of pixels of images that can be managed to:
         * CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK * MAX_GRID_SIZE (e.g. 1024*65535 on TESLA C-2075 GPU card)
         */

        /*
         * PARAMETERS
         */
        CUresult err;
        CUcontext context = null;
        CUmodule module = null;

        CUdeviceptr dev_BIN1 = null;
        CUdeviceptr dev_BIN2 = null;
        CUdeviceptr dev_ROI = null;
        CUdeviceptr dev_LTAKE_map = null;
        CUdeviceptr dev_LTAKE_count = null;

        // int gpuDeviceCount[] = { 0 };
        int elapsed_time = 0;
        int gpuDev = 0;
        // count the number of kernels that must print their output:
        int count_print = 0;

        try {
            int deviceCount = getGPUDeviceCount();
            /*
             * SELECT THE FIRST DEVICE (but I should select/split devices according to streams)
             */
            int selDev = getRandomGpuDevice(deviceCount);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, selDev);
            /*
             * QUERY CURRENT GPU PROPERTIES
             */
            int amountProperty[] = { 0 };
            // -1- threads / block
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            int maxThreadsPerBlock = amountProperty[0];
            // -2- blocks / grid
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
            int maxGridSize = amountProperty[0];
            // -3- No. Streaming Multiprocessors
            cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            int N_sm = amountProperty[0];
            // -4- threads / SM
            cuDeviceGetAttribute(amountProperty,
                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
            int max_threads_per_SM = amountProperty[0];

            /*
             * CREATE THE CONTEXT (for currently selected device)
             */
            context = new CUcontext();
            // int cuCtxCreate_STATUS =
            cuCtxCreate(context, selDev, device);
            // Load the ptx file:
            module = new CUmodule();
            cuModuleLoad(module, PTXFILE_land_take);
            /*
             * ENTRY POINTS (obtain a function pointer to the -2- entry functions/kernels)
             */
            // zero kernel
            CUfunction F_filter_roi = new CUfunction();
            cuModuleGetFunction(F_filter_roi, module, "_Z10filter_roiPhPKhj");
            // first kernel
            CUfunction F_histc_4 = new CUfunction();
            cuModuleGetFunction(F_histc_4, module,
                    "_Z32imperviousness_change_histc_sh_4PKhS0_jjPii");
            // second kernel
            CUfunction F_chmap = new CUfunction();
            // cuModuleGetFunction(F_chmap, module,
            // "_Z21imperviousness_changePKhS0_jjPi");
            cuModuleGetFunction(F_chmap, module, "_Z28imperviousness_change_doublePKhS0_jjPd");

            /*
             * DIM of ARRAYS in BYTES
             */
            int WIDTH = beans.get(admin_unit).width;
            int HEIGHT = beans.get(admin_unit).height;
            double areaOfOnePx = beans.get(admin_unit).getAreaPx();
            int map_len = WIDTH * HEIGHT;
            long sizeChar = map_len * Sizeof.BYTE;
            long sizeInt = numberOfBins * Sizeof.INT;
            long sizeDouble = map_len * Sizeof.DOUBLE;
            /*
             * EXIT( large images ) -map_len greater than available threads; -I should develop a new kernel in which each thread covers many pixels,
             * to deal with large images;
             */
            if (map_len > maxThreadsPerBlock * maxGridSize) {
                throw new ProcessException("Exceded the maximum number of pixels ("
                        + maxThreadsPerBlock * maxGridSize
                        + ") that the basic <<<imperviousness_change>>> kernel can handle!");
            }

            /*
             * CPU ARRAYS
             */
            // int host_LTAKE_map[] = new int[map_len];
            double host_LTAKE_map[] = new double[map_len];
            int host_LTAKE_count[] = new int[numberOfBins];
            // cuMemAllocHost( Pointer.to(host_LTAKE_map), sizeInt );
            cuMemAllocHost(Pointer.to(host_LTAKE_map), sizeDouble);
            cuMemAllocHost(Pointer.to(host_LTAKE_count), sizeInt);
            /*
             * GPU ARRAYS use CUDA_CHECK_RETURN() for all calls.
             */
            // get pointers
            dev_BIN1 = new CUdeviceptr();
            dev_BIN2 = new CUdeviceptr();
            dev_ROI = new CUdeviceptr();
            dev_LTAKE_map = new CUdeviceptr();
            dev_LTAKE_count = new CUdeviceptr();
            // allocate in mem
            cuMemAlloc(dev_BIN1, sizeChar);
            cuMemAlloc(dev_BIN2, sizeChar);
            cuMemAlloc(dev_ROI, sizeChar);
            // cuMemAlloc( dev_LTAKE_map, sizeInt );
            cuMemAlloc(dev_LTAKE_map, sizeDouble);
            cuMemAlloc(dev_LTAKE_count, sizeInt);
            // set mem
            cudaMemset(dev_LTAKE_count, 0, sizeInt);
            /*
             * MEM COPY H2D
             */
            cuMemcpyHtoD(dev_BIN1, Pointer.to(beans.get(admin_unit).getReferenceImage()), sizeChar);
            cuMemcpyHtoD(dev_BIN2, Pointer.to(beans.get(admin_unit).getCurrentImage()), sizeChar);
            cuMemcpyHtoD(dev_ROI, Pointer.to(beans.get(admin_unit).roi), sizeChar);

            /*
             * KERNELS GEOMETRY NOTE: use ceil() instead of the "%" operator!!!
             */
            int bdx, gdx, gdx_2, num_blocks_per_SM, mapel_per_thread, Nblks_per_grid, sh_mem;
            bdx = BLOCKDIM;
            num_blocks_per_SM = max_threads_per_SM / bdx;
            mapel_per_thread = (int) Math.ceil((double) map_len
                    / (double) ((bdx * 1) * N_sm * num_blocks_per_SM));
            gdx = (int) Math.ceil((double) map_len / (double) (mapel_per_thread * (bdx * 1)));
            gdx_2 = (int) Math.ceil((double) map_len / (double) ((bdx * numberOfBins)));
            Nblks_per_grid = N_sm * (max_threads_per_SM / threads_lt);
            sh_mem = (bdx * numberOfBins) * Sizeof.INT;
            /*
             * KERNELS INVOCATION
             * 
             * ****************************** -1- imperviousness_change_hist -2- imperviousness_change ******************************
             * 
             * Note that imperviousness_change_large does not work!!
             */

            // ***00***
            // filter_roi<<<dimGrid,dimBlock>>>(dev_BIN,dev_ROI,map_len);
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_00 = Pointer.to(Pointer.to(dev_BIN1), Pointer.to(dev_ROI),
                    Pointer.to(new int[] { map_len }));
            // Call the kernel function.
            cuLaunchKernel(F_filter_roi, Nblks_per_grid, 1, 1, // Grid dimension ::
                                                               // dimGrid(
                                                               // Nblks_per_grid,
                                                               // 1, 1 )
                    threads_lt, 1, 1, // Block dimension :: dimBlock( threads_ccl,
                                      // 1, 1 )
                    0, null, // Shared memory size and stream
                    kern_00, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            kern_00 = Pointer.to(Pointer.to(dev_BIN2), Pointer.to(dev_ROI),
                    Pointer.to(new int[] { map_len }));
            // Call the kernel function.
            cuLaunchKernel(F_filter_roi, Nblks_per_grid, 1, 1, // Grid dimension ::
                                                               // dimGrid(
                                                               // Nblks_per_grid,
                                                               // 1, 1 )
                    threads_lt, 1, 1, // Block dimension :: dimBlock( threads_ccl,
                                      // 1, 1 )
                    0, null, // Shared memory size and stream
                    kern_00, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            // end_t = clock();
            // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
            // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
            // elapsed_time += (int)( (double)(end_t - start_t ) /
            // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

            // ***01***
            // imperviousness_change_histc_sh_4<<<grid,block,sh_mem>>>(dev_BIN1,
            // dev_BIN2, width, heigth, dev_LTAKE_count, mapel_per_thread );
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_hist4 = Pointer.to(Pointer.to(dev_BIN1), Pointer.to(dev_BIN2),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(dev_LTAKE_count), Pointer.to(new int[] { mapel_per_thread }));
            // Call the kernel function.
            cuLaunchKernel(F_histc_4, gdx, 1, 1, // Grid dimension (gdx,1,1)
                    bdx, 1, 1, // Block dimension (bdx,1,1)
                    sh_mem, null, // Shared memory size and stream
                    kern_hist4, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            // end_t = clock();
            // System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)(
            // (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
            // elapsed_time += (int)( (double)(end_t - start_t ) /
            // (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

            // ***02***
            // imperviousness_change_double<<<grid_2,block_2>>>(dev_BIN1,dev_BIN2,width,heigth,dev_LTAKE_map);
            // start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which
            // point to the actual values.
            Pointer kern_chmap = Pointer.to(Pointer.to(dev_BIN1), Pointer.to(dev_BIN2),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(dev_LTAKE_map));
            // Call the kernel function.
            cuLaunchKernel(F_chmap, gdx_2, 1, 1, // Grid dimension (gdx_2,1,1)
                    bdx * numberOfBins, 1, 1, // Block dimension (bdx *
                                              // numberOfBins, 1, 1)
                    0, null, // Shared memory size and stream
                    kern_chmap, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            /*
             * end_t = clock(); System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print ,kern_24,(int)( (double)(end_t - start_t ) /
             * (double)CLOCKS_PER_SEC * 1000 )); elapsed_time += (int)( (double)(end_t - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time
             * [ms]:
             */

            /*
             * System.out.println("______________________________________\n"); System.out.println("  %21s\t%6d [msec]\n",
             * "Total time (T):",elapsed_time );
             */

            // cuMemcpyDtoH(Pointer.to(host_LTAKE_map), dev_LTAKE_map, sizeInt );
            cuMemcpyDtoH(Pointer.to(host_LTAKE_map), dev_LTAKE_map, sizeDouble);
            cuMemcpyDtoH(Pointer.to(host_LTAKE_count), dev_LTAKE_count, sizeInt);

            /*
             * long estimatedTime = System.currentTimeMillis() - startTime; System.out.println("Elapsed time fragmentation()" + estimatedTime +
             * " [ms]");
             */
            // conversion(int --> double) & transform(count --> area)
            double outCount[] = new double[numberOfBins];
            for (int ii = 0; ii < numberOfBins; ii++) {
                outCount[ii] = ((double) host_LTAKE_count[ii] * areaOfOnePx) * m2_to_ha; // take only the net value!!
            }

            if (UrbanGridCUDAProcess.TESTING) {
                try {
                    UrbanGridCUDAProcess.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT,
                            host_LTAKE_map, DataBuffer.TYPE_DOUBLE, "ssgci_LTAKE_map");
                    storePlainTextSampleFile_double(outCount, "ssgci_LTAKE_count");
                } catch (IOException e) {
                    LOGGER.log(Level.WARNING, "Could not save GeoTIFF Sample for testing", e);
                }
            }

            // System.out.println("\n CUDA Finished!!\n");
            return Arrays.asList(host_LTAKE_map, outCount);

        } finally {
            try {
                // CUDA free:
                if (dev_BIN1 != null)
                    cuMemFree(dev_BIN1);
                if (dev_BIN2 != null)
                    cuMemFree(dev_BIN2);
                if (dev_LTAKE_map != null)
                    cuMemFree(dev_LTAKE_map);
                if (dev_LTAKE_count != null)
                    cuMemFree(dev_LTAKE_count);
                if (dev_ROI != null)
                    cuMemFree(dev_ROI);

                // Unload MODULE
                if (module != null)
                    cuModuleUnload(module);

                // Destroy CUDA context:
                if (context != null)
                    cuCtxDestroy(context);

            } catch (Exception e) {
                // insert ...
            }
        }
    }

    /**
     * 
     * @param beans :: list of layers and their properties
     * @param admin_unit :: administrative unit being processed
     * @return No. of Adults per year that cannot be fed anymore due to loss of wheat equivalent food supply
     */
    public static double potloss_foodsupply(List<CUDABean> beans, int admin_unit) {
        final List<double[]> res_LT = CUDAClass.land_take(beans, admin_unit);

        double[] LTAKE_count = res_LT.get(1);// account in hectares
        /*
         * ----------------------------- (N) (BIN2,BIN1) --> (LTAKE) ----------------------------- (1) (0,0) --> +0 ---> nothing changed in rural
         * pixels LTAKE_count[0] (2) (0,1) --> -1 ---> increase of rural pixels LTAKE_count[1] (3) (1,0) --> +1 ---> increase of urban pixels
         * LTAKE_count[2] (4) (1,1) --> -0 ---> nothing changed in urban pixels LTAKE_count[3] -----------------------------
         */

        // ( [ha] - [ha] ) * [persons * year-1 * ha-1] ––> [persons * year-1]
        double wheatLoss = (LTAKE_count[2] - LTAKE_count[1]) * fedPersons;// [persons
                                                                          // *
                                                                          // year-1]
        /*
         * wheatLoss > 0 ––> loss of wheat supply wheatLoss = 0 ––> unchanged wheat supply wheatLoss < 0 ––> gain of wheat supply
         */

        if (UrbanGridCUDAProcess.TESTING) {
            try {
                storeIndex_double(wheatLoss, "ssgci__10");
            } catch (IOException e) {
                LOGGER.log(Level.WARNING,
                        "Could not save Index[10] 'Loss of Food Supply' for testing", e);
            }
        }

        return wheatLoss;
    }

    public static double[] newUrbanization(List<CUDABean> beans, boolean rural, int ray_pixels,
            int i, int j) {
        // int j = 0; // fictitious admin unit
        // int i = 0; // fictitious year
        int WIDTH = beans.get(j).width;
        int HEIGHT = beans.get(j).height;
        int map_len = WIDTH * HEIGHT;
        double[] result = new double[map_len];

        // HERE I HAVE TO CREATE THE LAYER ACCOUNTING THE NEW URBANIZATION
        // IT IS ASSUMED THAT FRAGMENTATION IS RUN USING ONE YEAR, HENCE
        // WHEN I LAUNCH newUrbanization process I HAVE THE FOLLOWING CONDITIONS:
        // a) rural was already configured and is stored in beans yet;
        // b) ray_pixels is already available in the ENV;
        // c) the fictitious reference is reference;
        // d) the fictitious current is reference + bin ––> (0+0=0; 0+1=1; 1+0=1; 1+1=1)
        // NOW IF I COMPUTE newUrbanization using
        // i. fragmentation ––> I ONLY NEED fictitious current;
        // ii.land take ––> I USE BOTH FICTITIOUS LAYERS (reference & current).
        // AT NOW I CONSIDER (i) process!

        // I need to write in beans the fictitious current:
        // [create a JCuda class instead]
        byte[] fictitious_curImage = new byte[map_len];
        byte[] fictitious_refImage = beans.get(j).getReferenceImage();
        byte[] ROI = beans.get(j).roi;

        for (int ii = 0; ii < map_len; ii++) {
            fictitious_curImage[ii] = (byte)(fictitious_refImage[ii] + (ROI[ii]>0?0:1));
        }
        beans.get(j).setCurrentImage(fictitious_curImage);

        // –Land Take–
        // final List<double[]> resultCuda = CUDAClass.land_take( beans, j );
        // result[j][i] = resultCuda.get(0);// MAP

        // –Fragmentation–
        i = 1;// I run fragmentation on fictitious current which is at current layer position!
        boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
        if (isFeasible)
            result = CUDAClass.fragmentation(beans, rural, ray_pixels, i, j);
        else
            result = new double[] { Double.NaN };
        return result;
    }

    public static int getRandomGpuDevice(int deviceCount) {
        Random rn = new Random();
        return rn.nextInt(deviceCount);
    }

}// close :: public class CUDAClass
