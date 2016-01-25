package org.geoserver.wps.gs.soilsealing;
import java.util.Arrays;
import java.util.List;

import org.geoserver.wps.gs.soilsealing.UrbanGridCUDAProcess.CUDABean;
import org.geotools.process.ProcessException;

//import java.util.Date;





import jcuda.*;
import jcuda.runtime.*;
import jcuda.driver.*;
import static jcuda.driver.CUresult.*;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
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

public class CUDAClass {
	
	private final static String PTXFILE_ccl				= "/opt/soil_sealing/cudacodes/ccl.ptx";
	private final static String PTXFILE_fragmentation 	= "/opt/soil_sealing/cudacodes/fragmentation.ptx";
	private final static String PTXFILE_land_take		= "/opt/soil_sealing/cudacodes/land_take.ptx";
	
//	private Clock			start_t,end_t;
	
	/*		SIZE of BLOCK
	 * 		to set dynamic use instead: floor( sqrt( maxThreadsPerBlock ))
	 * 		ATTENTION: this value and the one set in .cu file used to compile .ptx must be the same !!!!!
	 */
	private final static int BLOCKDIM_X = 32;
	private final static int BLOCKDIM 	= 256;

/*	public static void CUDA_CHECK_RETURN(int value){  
		if ( value != 0 ) {
			System.out.println(stderr, "Error %s at line %d in file %s\n",
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);
			exit(1);
		}
	}
*/	
	/*
	#define CUDA_CHECK_RETURN(err)  __checkCudaErrors (err, __FILE__, __LINE__)

	inline void __checkCudaErrors( CUresult err, const char *file, const int line )
	{
	    if( CUDA_SUCCESS != err) {
	        fprintf(stderr,
	                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
	                err, file, line );
	        exit(-1);
	    }
	}
	*/


	/**
	 * This function calculates the rural/urbal fragmentation according to the user-defined radius.
	 * @param beans :: list of layers and their properties 
	 * @param rural :: flag for rural(=TRUE)/urban(=FALSE) switch
	 * @param areaOfOnePixel :: pixel area in m2
	 * @param RADIUS :: radius of counting as number of pixels
	 * @param year :: index defining the year in which calculate fragmentation (reference, current)
	 * @param admin_unit :: index defining the administrative unit which is undergoing calculation
	 * @return 
	 */
	public static double[] fragmentation(
											List<CUDABean> beans,
											boolean rural, double areaOfOnePixel, int RADIUS,
											int year,int admin_unit
										){
        /**
         * 	Copyright 2015 Giuliano Langella: ---- working fine!! ----
         * 	
         * 	This function computes the fragmentation in driver-API CUDA-C.
         * 	-1- complementary_to_ONE	{ BIN,ONE		} --> COMP
         * 	-2-	gtranspose				{ BIN			}
         * 	-3- Vcumsum					{ BIN			} --> FRAG
         * 	-4- sum_of_3_LINES			{ FRAG			}
         * 	-5- gtranspose				{ FRAG			}
         * 	-6- Vcumsum					{ FRAG			}
         * 	-7- sum_of_3_LINES			{ FRAG			}
         * 	-8- mask_twice				{ FRAG,COMP,ROI	} --> FRAG
         */

		/*
		 * 	NOTES
		 *  I should use a more performing transpose algorithm (maybe based on shared mem).
		 *  The best way to deal with more administrative units is to use cuda streams in parallel.
		 *  At now, for simplicity I assume that the fragmentation JCuda class is called for any
		 *  admin unit from outside. But to allow an intermediate implementation, I pass all the 
		 *  admin units to the function, and I extract the usefull info for the first admin unit.
		 *  After I have to enlarge the code for usign streams.
		 */

//		long startTime = System.currentTimeMillis();

		/*
		 * 		PARAMETERS
		 */
		CUresult err;
		int mask_len 			= RADIUS*2+1;
		double std_4_area 		= mask_len*mask_len; //(RADIUS*RADIUS*areaOfOnePixel);
		int gpuDeviceCount[]	= { 0 };
    	int elapsed_time		= 0;
    	// count the number of kernels that must print their output:
    	int count_print 		= 0;
//    	System.out.println("Remember that you have to activate the division by std_4_area in the cuda kernel (mask_twice)!!\n");
    	//rural = true; System.out.println("delete 'rural = true' from code.");
    	
		/*
		 * 		ESTABILISH CONTEXT
		 */
		JCudaDriver.setExceptionsEnabled(true);
        // Initialise the driver:
		//err = 
		cuInit(0);

        /*
         *  	RECOGNIZE DEVICE(s) EXISTENCE:
         */
/*	    if (err == CUDA_SUCCESS)
	    	CUDA_CHECK_RETURN(cuDeviceGetCount(gpuDeviceCount));
	 if (deviceCount == 0) {
	        System.out.println("Error: no devices supporting CUDA\n");
	        exit(-1);
	    }
*/	    
		// Obtain the number of devices
        cuDeviceGetCount(gpuDeviceCount);
        int deviceCount = gpuDeviceCount[0];
        if (deviceCount == 0) {
            throw new ProcessException("Error: no devices supporting CUDA.");
        }
        
        /*
         *		SELECT THE FIRST DEVICE
         *		(but I should select/split devices according to streams)
         */
        int selDev = 0;
        CUdevice device = new CUdevice();
        cuDeviceGet(device, selDev);
        // Query some useful properties:
        int amountProperty[] = { 0 };
        // -1-
        cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        int maxThreadsPerBlock = amountProperty[0];
        // -2-
        //...others?

        /*
         * 		CREATE THE CONTEXT
         * 		(for currently selected device)
         */
        CUcontext context = new CUcontext();
        // int cuCtxCreate_STATUS =
        cuCtxCreate(context, selDev, device);
        // Load the ptx file:
        CUmodule module = new CUmodule();
        cuModuleLoad(module, PTXFILE_fragmentation);
        /*
         *  	ENTRY POINTS
         *  	(obtain a function pointer to
         *  	 the -5- entry functions/kernels)
         */
        // complementary_to_ONE
        CUfunction F_complementary_to_ONE = new CUfunction();
        cuModuleGetFunction(F_complementary_to_ONE, module, "_Z20complementary_to_ONEPKhS0_Phjj");
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
         *  	DIM of ARRAYS in BYTES
         */
        int WIDTH 			= beans.get(admin_unit).width;
        int HEIGHT			= beans.get(admin_unit).height;
    	int map_len 		= WIDTH * HEIGHT;
    	long sizeChar		= map_len*Sizeof.BYTE;
    	long sizeDouble		= map_len*Sizeof.DOUBLE;
    	/*
    	 * 		CPU ARRAYS
    	 */
    	double 	host_FRAG[] = new double[map_len];
    	double 	host_IO[] 	= new double[map_len];
    	byte 	host_TMP[]	= new byte	[map_len];
    	byte 	host_COMP[] = new byte	[map_len];
    	cuMemAllocHost( 	Pointer.to(host_FRAG), 	sizeDouble	);
    	cuMemAllocHost( 	Pointer.to(host_IO), 	sizeDouble	);
    	cuMemAllocHost( 	Pointer.to(host_TMP),	sizeChar	);
    	cuMemAllocHost( 	Pointer.to(host_COMP), 	sizeChar	);
    	/*
    	 * 		GPU ARRAYS
    	 * 		use CUDA_CHECK_RETURN() for all calls.
    	 */
    	// get pointers
    	CUdeviceptr dev_FRAG 	= new CUdeviceptr();
    	CUdeviceptr dev_IO 		= new CUdeviceptr();
    	CUdeviceptr dev_BIN 	= new CUdeviceptr();
    	CUdeviceptr dev_ROI 	= new CUdeviceptr();
    	CUdeviceptr dev_TMP 	= new CUdeviceptr();
    	CUdeviceptr dev_COMP 	= new CUdeviceptr();
    	CUdeviceptr dev_ONE 	= new CUdeviceptr();
    	// allocate in mem
    	cuMemAlloc(dev_FRAG, 	sizeDouble);
    	cuMemAlloc(dev_IO, 		sizeDouble);
        cuMemAlloc(dev_BIN, 	sizeChar);
        cuMemAlloc(dev_ROI, 	sizeChar);
        cuMemAlloc(dev_TMP, 	sizeChar);
        cuMemAlloc(dev_COMP, 	sizeChar);
        cuMemAlloc(dev_ONE, 	sizeChar);
        // set mem
    	cudaMemset(	dev_FRAG,	0,  sizeDouble	);
    	cudaMemset(	dev_IO,		0,  sizeDouble	);
    	cudaMemset(	dev_COMP,	0,  sizeChar	);
    	cudaMemset(	dev_ONE,	1,	sizeChar	);
        /*
         * 		MEM COPY H2D
         */
//    	double[][] results = new double[year][map_len];
    	if (year == 0) {
        	cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getReferenceImage()),sizeChar);
        }else if(year == 1) {
        	cuMemcpyHtoD(dev_BIN, Pointer.to(beans.get(admin_unit).getCurrentImage()),	sizeChar);
        }else if(year == 2){
        	// do nothing because the DIFF is calculated in JAVA
        	
/*        	System.out.println("You should implement in CUDA the difference between ref & curr!!");
        	for(int ii=0;ii<map_len;ii++){
        		host_FRAG[ii] = beans.get(admin_unit).referenceImage[ii]-beans.get(admin_unit).getCurrentImage()[ii];
        	}
        	return host_FRAG;
*/      }
        cuMemcpyHtoD(dev_ROI, Pointer.to(beans.get(admin_unit).roi), 					sizeChar);

    	/*
    	 * 		KERNELS GEOMETRY
    	 * 		NOTE: use ceil() instead of the "%" operator!!!
    	 */
    	int BDX = BLOCKDIM_X; //
    	int 	gdx_k12, gdy_k12, gdx_k3, gdy_k3, gdx_trans, gdy_trans, gdx_k12_t, gdy_k12_t,gdx_mask,gdy_mask;
    	// k1 + k2
    	gdx_k12 	= (((WIDTH  % mask_len)>0)? 1:0) 	+ (WIDTH  / mask_len);
    	gdy_k12 	= (((HEIGHT % (BDX*BDX))>0)? 1:0) 	+ (int)Math.floor(HEIGHT / (BDX*BDX));
    	// k3 + k4
    	gdx_k3 		= (((WIDTH  % (BDX*BDX))>0)? 1:0) 	+ (WIDTH  / (BDX*BDX));
    	gdy_k3 		= (((HEIGHT % mask_len)>0)? 1:0) 	+ (int)Math.floor(HEIGHT / mask_len);
    	gdx_k12_t 	= (((HEIGHT % (BDX*BDX))>0)? 1:0) 	+ (HEIGHT  / (BDX*BDX));
    	gdy_k12_t 	= (((WIDTH  % mask_len)>0)? 1:0) 	+ (int)Math.floor(WIDTH / mask_len);
    	// k(gtransform)
    	gdx_trans 	= (((WIDTH  % BDX)>0)? 1:0) 		+ WIDTH  / BDX;
    	gdy_trans 	= (((HEIGHT % BDX)>0)? 1:0) 		+ HEIGHT / BDX;
    	// mask_twice
    	gdx_mask	= (((WIDTH  % BDX)>0)? 1:0) 		+ WIDTH  / BDX;
    	gdy_mask 	= (((HEIGHT % BDX)>0)? 1:0) 		+ HEIGHT / BDX;

    	/*
    	 * 	ALTERNATIVE ALGORITHM
    	 * 		TRY using matrix transpose to use the cumsum_vertical & sum_of_3_rows twice:
    	 * 		once for step 3 & 4 as regularly done, and the other one for step 1 & 2 in
    	 * 		place of kernels that are too slow (cumsum_horizontal & sum_of_3_cols).
    	 *
    	 * 		Speed tests demonstrate that working along Y when doing cumulative sum
    	 * 		is 10 times more efficient: this is true because warps of threads R/W in
    	 * 		coalesced patterns.
    	 *
    	 * 		Try to apply ROI at the end, so that I skip one gtranspose at the beginning.
    	 *
    	 */
//		complementary_to_ONE<unsigned char><<<grid_compl,block_compl>>>( dev_ONE,dev_BIN, dev_COMP, MDbin.width, MDbin.heigth );
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_compl = Pointer.to(
            Pointer.to(dev_ONE), Pointer.to(dev_BIN), Pointer.to(dev_COMP),
            Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT })
        );

        // Call the kernel function.
        cuLaunchKernel(F_complementary_to_ONE,
    		gdx_mask, gdy_mask, 1,	// Grid dimension
    		BDX, BDX, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_compl, null		// Kernel- and extra parameters
        );
        cuCtxSynchronize();        
//    	end_t = clock();
//    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
//    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

    	if(rural==true){
    		/**
    		 * 	This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
    		 * 		FRAG = fragmentation_prog( BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
    		 * 		FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
    		 * 	* 	This means that the use of BIN & COMP is straightforward:
    		 */
//    		gtranspose_char<<<grid_trans,block_trans>>>( dev_TMP2, dev_BIN, MDbin.width, MDbin.heigth );
//    		start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
            Pointer kern_gtrans = Pointer.to(
                Pointer.to(dev_TMP), Pointer.to(dev_BIN),
                Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT })
            );
            // Call the kernel function.
            cuLaunchKernel(F_gtranspose_char,
        		gdx_trans, gdy_trans, 1,// Grid dimension
        		BDX, BDX, 1,			// Block dimension
                0, null,				// Shared memory size and stream
                kern_gtrans, null		// Kernel- and extra parameters
            );
            cuCtxSynchronize();
//    		end_t = clock();
//    		System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
//    		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
    	}
    	else if(rural==false){
    		/**
    		 * 	This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
    		 * 		FRAG = fragmentation_prog( COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
    		 * 		FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
    		 * 	This means that I have to invert BIN & COMP:
    		 */
//    		gtranspose_char<<<grid_trans,block_trans>>>( dev_TMP2, dev_BIN, MDbin.width, MDbin.heigth );
//    		start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
            Pointer kern_gtrans = Pointer.to(
                Pointer.to(dev_TMP), Pointer.to(dev_COMP),
                Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT })
            );
            // Call the kernel function.
            cuLaunchKernel(F_gtranspose_char,
        		gdx_trans, gdy_trans, 1,// Grid dimension
        		BDX, BDX, 1,			// Block dimension
                0, null,				// Shared memory size and stream
                kern_gtrans, null		// Kernel- and extra parameters
            );
            cuCtxSynchronize();
/*    		end_t = clock();
    		System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans_,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/    	}

//		Vcumsum_char<<<grid_k12_t,block_k12_t>>>( dev_TMP2, MDbin.heigth,MDbin.width,dev_FRAG,RADIUS );    	
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_vsum = Pointer.to(
            Pointer.to(dev_TMP),
            Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH }),
            Pointer.to(dev_FRAG),
            Pointer.to(new int[] { RADIUS })
        );
        // Call the kernel function.
        cuLaunchKernel(F_Vcumsum_char,
    		gdx_k12_t, gdy_k12_t, 1,// Grid dimension
    		BDX*BDX, 1, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_vsum, null			// Kernel- and extra parameters
        );
        cuCtxSynchronize();
/*     	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_13,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/    	
//		sum_of_3_LINES<<<grid_k12_t,block_k12_t>>>( 	dev_FRAG, MDbin.heigth, MDbin.width, dev_IO, RADIUS 		);    	
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_s3L = Pointer.to(
            Pointer.to(dev_FRAG),
            Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH }),
            Pointer.to(dev_IO),
            Pointer.to(new int[] { RADIUS })
        );
        // Call the kernel function.
        cuLaunchKernel(F_sum_of_3_LINES,
    		gdx_k12_t, gdy_k12_t, 1,// Grid dimension
    		BDX*BDX, 1, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_s3L, null			// Kernel- and extra parameters
        );
        cuCtxSynchronize();
//    	sum_of_3_LINES<<<grid_k12_t,block_k12_t>>>( 	dev_FRAG, MDbin.heigth, MDbin.width, dev_IO, RADIUS 		);
/*    	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_24,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/
//    	gtranspose_double<<<grid_trans2,block_trans>>>(dev_FRAG, dev_IO, MDbin.heigth, MDbin.width);
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_gtrans = Pointer.to(
            Pointer.to(dev_FRAG), Pointer.to(dev_IO),
            Pointer.to(new int[] { HEIGHT }), Pointer.to(new int[] { WIDTH })
        );
        // Call the kernel function.
        cuLaunchKernel(F_gtranspose_double,
    		gdy_trans, gdx_trans, 1,// Grid dimension
    		BDX, BDX, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_gtrans, null		// Kernel- and extra parameters
        );
        cuCtxSynchronize();
/*    	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/
//    	Vcumsum_double<<<grid_k3,block_k3>>>( dev_FRAG, MDbin.width,MDbin.heigth,dev_IO,RADIUS ); // { ",unsigned char" ; "dev_ROI, " }
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_vsum2 = Pointer.to(
            Pointer.to(dev_FRAG),
            Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
            Pointer.to(dev_IO),
            Pointer.to(new int[] { RADIUS })
        );
        // Call the kernel function.
        cuLaunchKernel(F_Vcumsum_double,
    		gdx_k3,gdy_k3, 1,		// Grid dimension
    		BDX*BDX, 1, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_vsum2, null			// Kernel- and extra parameters
        );
        cuCtxSynchronize();
/*    	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_13,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/
//    	sum_of_3_LINES<<<grid_k3,block_k3>>>( 	dev_IO, MDbin.width, MDbin.heigth, dev_FRAG, RADIUS 		);
//    	start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_s3L2 = Pointer.to(
            Pointer.to(dev_IO),
            Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
            Pointer.to(dev_FRAG),
            Pointer.to(new int[] { RADIUS })
        );
        // Call the kernel function.
        cuLaunchKernel(F_sum_of_3_LINES,
    		gdx_k3,gdy_k3, 1,		// Grid dimension
    		BDX*BDX, 1, 1,			// Block dimension
            0, null,				// Shared memory size and stream
            kern_s3L2, null			// Kernel- and extra parameters
        );
        cuCtxSynchronize();
/*    	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_24,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/
    	if(rural==true){
    		/**
    		 * 	This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
    		 * 		FRAG = fragmentation_prog( BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
    		 * 		FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
    		 * 	* 	This means that the use of BIN & COMP is straightforward:
    		 */
//    		mask_twice<<<grid_mask,block_mask>>>( 	dev_FRAG, dev_ROI, dev_COMP, MDbin.width, MDbin.heigth, std_4_area	);
//    		start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
            Pointer kern_m2 = Pointer.to(
                Pointer.to(dev_FRAG), Pointer.to(dev_ROI), Pointer.to(dev_COMP),
                Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
                Pointer.to(new double[] { std_4_area })
            );
            // Call the kernel function.
            cuLaunchKernel(F_mask_twice,
        		gdx_mask, gdy_mask, 1,	// Grid dimension
        		BDX, BDX, 1,			// Block dimension
                0, null,				// Shared memory size and stream
                kern_m2, null			// Kernel- and extra parameters
            );
            cuCtxSynchronize();
/*    		end_t = clock();
    		System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_mask,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/    	}
    	else { // if(rural==false){
    		/**
    		 * 	This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
    		 * 		FRAG = fragmentation_prog( COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
    		 * 		FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
    		 * 	This means that I have to invert BIN & COMP:
    		 */
//    		mask_twice<<<grid_mask,block_mask>>>( 	dev_FRAG, dev_ROI, dev_BIN, MDbin.width, MDbin.heigth, std_4_area	);
//    		start_t = clock();
            // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
            Pointer kern_m2 = Pointer.to(
                    Pointer.to(dev_FRAG), Pointer.to(dev_ROI), Pointer.to(dev_BIN),
                    Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
                    Pointer.to(new double[] { std_4_area })
                );
            // Call the kernel function.
            cuLaunchKernel(F_mask_twice,
        		gdx_mask, gdy_mask, 1,	// Grid dimension
        		BDX, BDX, 1,			// Block dimension
                0, null,				// Shared memory size and stream
                kern_m2, null			// Kernel- and extra parameters
            );
            cuCtxSynchronize();
/*            end_t = clock();
    		System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_mask,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/    	}

/*    	System.out.println("______________________________________\n");
    	System.out.println("  %21s\t%6d [msec]\n", "Total time (T):",elapsed_time );
*/

    	// ...to be completed:
    	cuMemcpyDtoH(Pointer.to(host_FRAG), dev_FRAG, sizeDouble );

    	// CUDA free:
    	cuMemFree( dev_BIN	);
    	cuMemFree( dev_FRAG	);
    	cuMemFree( dev_ROI	);
    	cuMemFree( dev_TMP	);
    	cuMemFree( dev_IO	);

        // Unload MODULE
        cuModuleUnload(module);

        // Destroy CUDA context:
        cuCtxDestroy(context);

    	//System.out.println("\n CUDA Finished!!\n");
/*		long estimatedTime = System.currentTimeMillis() - startTime;
		System.out.println("Elapsed time fragmentation()" + estimatedTime + " [ms]");
*/
    	return host_FRAG; // --> return host_FRAG
	}	
/*
	private static long[] connected_component_labeling( byte URBAN, byte ROI ){
		return 0;
	}
	
	public static double[] urban_sprawl(List<CUDABean> beans,double areaOfOnePixel,int year,int admin_unit){
		
		return 0.0;
	}
*/

	/**
	 * 
	 * @param beans :: list of layers and their properties
	 * @param areaOfOnePixel :: pixel area
	 * @param admin_unit :: index defining the administrative unit being processed
	 * @return a list of two double objects{map,area}
	 */
	public static List<double[]> land_take(	List<CUDABean> beans,
										double areaOfOnePixel, int admin_unit) {
        /**
         * 	Copyright 2015 Giuliano Langella: ---- testing... ----
         * 	
         * 	This function computes the reduction kernel in driver-API CUDA-C.
         * 	-1- reduction	{ BIN1,BIN2,ROI } --> LTAKE_count 	[kernel::imperviousness_change_histc_sh_4]
         *  -2- difference	{ BIN1,BIN2,ROI } --> LTAKE_map 	[kernel::imperviousness_change]
         *  
         *  -----------------------------
		 *	(N) (BIN2,BIN1)	--> (LTAKE)
		 *	-----------------------------
		 *	(1) (0,0) 		--> +0			---> nothing changed in rural pixels	LTAKE_count[0]
		 *	(2) (0,1) 		--> -1			---> increase of rural pixels			LTAKE_count[1]
		 *	(3) (1,0) 		--> +1			---> increase of urban pixels			LTAKE_count[2]
		 *	(4) (1,1) 		--> -0			---> nothing changed in urban pixels	LTAKE_count[3]
		 *	-----------------------------
		 *	where values can be { 0:rural; 1:urban }.
         */

		/*
		 * 	NOTES
		 *  	1.\ The imperviousness_change_large kernel does not work! So I have to limit 
		 *  		the number of pixels of images that can be managed to:
		 *  		  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK * MAX_GRID_SIZE
		 *  		  (e.g. 1024*65535 on TESLA C-2075 GPU card)
		 */
		
		/*
		 * 	PARAMETERS
		 */
		CUresult err;
		int gpuDeviceCount[]	= { 0 };
	    int elapsed_time		= 0;
		int	gpuDev				= 0;
		// count the number of kernels that must print their output:
		int count_print 		= 0;
		
		/*
		 * 	ESTABILISH CONTEXT
		 */
		JCudaDriver.setExceptionsEnabled(true);
		// Initialise the driver:
		//CUresult err = 
		cuInit(0);

        /*
         *  	RECOGNIZE DEVICE(s) EXISTENCE:
         */
		// Obtain the number of devices
        cuDeviceGetCount(gpuDeviceCount);
        int deviceCount = gpuDeviceCount[0];
        if (deviceCount == 0) {
            throw new ProcessException("Error: no devices supporting CUDA.");
        }
        /*
         *	SELECT THE FIRST DEVICE
         *	(but I should select/split devices according to streams)
         */
        int selDev 			= 0;
        CUdevice device 	= new CUdevice();
        cuDeviceGet(device, selDev);
    	/*
    	 * 	QUERY CURRENT GPU PROPERTIES
    	 */
        int amountProperty[] 	= { 0 };
        // -1- threads / block
        cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        int maxThreadsPerBlock 	= amountProperty[0];
        // -2- blocks / grid
        cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
        int maxGridSize			= amountProperty[0];
        // -3- No. Streaming Multiprocessors
        cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        int N_sm				= amountProperty[0];
        // -4- threads / SM
        cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    	int max_threads_per_SM	= amountProperty[0];
    	
        /*
         * 	CREATE THE CONTEXT
         * 	(for currently selected device)
         */
        CUcontext context = new CUcontext();
        // int cuCtxCreate_STATUS =
        cuCtxCreate(context, selDev, device);
        // Load the ptx file:
        CUmodule module = new CUmodule();
        cuModuleLoad(module, PTXFILE_land_take);
        /*
         *  ENTRY POINTS
         *  (obtain a function pointer to
         *   the -2- entry functions/kernels)
         */
        // first kernel
        CUfunction F_histc_4 = new CUfunction();
        cuModuleGetFunction(F_histc_4, module, "_Z32imperviousness_change_histc_sh_4PKhS0_jjPii");
        // second kernel
        CUfunction F_chmap = new CUfunction();
        //cuModuleGetFunction(F_chmap, module, "_Z21imperviousness_changePKhS0_jjPi");
        cuModuleGetFunction(F_chmap, module, "_Z28imperviousness_change_doublePKhS0_jjPd");

        /*
         *  	DIM of ARRAYS in BYTES
         */
        int WIDTH 		= beans.get(admin_unit).width;
        int HEIGHT		= beans.get(admin_unit).height;
    	int map_len 	= WIDTH * HEIGHT;
    	long sizeChar	= map_len*Sizeof.BYTE;
    	long sizeInt	= map_len*Sizeof.INT;
    	long sizeDouble	= map_len*Sizeof.INT;
    	/*
    	 * 	EXIT( large images )
    	 * 		-map_len greater than available threads;
    	 * 		-I should develop a new kernel in which each thread covers many pixels, to deal with large images; 
    	 */
    	if (map_len > maxThreadsPerBlock * maxGridSize){
    		throw new ProcessException("Exceded the maximum number of pixels (" + maxThreadsPerBlock * maxGridSize + ") that the basic <<<imperviousness_change>>> kernel can handle!");
    	}

    	/*
    	 *	CPU ARRAYS
    	 */
    	//int 	host_LTAKE_map[] 	= new int[map_len];
    	double 	host_LTAKE_map[] 	= new double[map_len];
    	int 	host_LTAKE_count[]	= new int[4];
    	//cuMemAllocHost( 	Pointer.to(host_LTAKE_map), 	sizeInt		);
    	cuMemAllocHost( 	Pointer.to(host_LTAKE_map), 	sizeDouble	);
    	cuMemAllocHost( 	Pointer.to(host_LTAKE_count),	sizeInt		);
    	/*
    	 * 	GPU ARRAYS
    	 * 	use CUDA_CHECK_RETURN() for all calls.
    	 */    	
    	// get pointers
    	CUdeviceptr dev_BIN1			= new CUdeviceptr();
    	CUdeviceptr dev_BIN2 			= new CUdeviceptr();
    	CUdeviceptr dev_ROI 			= new CUdeviceptr();
    	CUdeviceptr dev_LTAKE_map 		= new CUdeviceptr();
    	CUdeviceptr dev_LTAKE_count 	= new CUdeviceptr();
    	// allocate in mem
        cuMemAlloc( dev_BIN1, 			sizeChar	);
        cuMemAlloc( dev_BIN2, 			sizeChar	);
        cuMemAlloc( dev_ROI, 			sizeChar	);
        //cuMemAlloc( dev_LTAKE_map, 		sizeInt		);
        cuMemAlloc( dev_LTAKE_map, 		sizeDouble	);
        cuMemAlloc( dev_LTAKE_count,	sizeInt		);
        // set mem
    	cudaMemset(	dev_LTAKE_count,0,  4*Sizeof.INT);
        /*
         * 	MEM COPY H2D
         */
    	cuMemcpyHtoD(dev_BIN1, Pointer.to(beans.get(admin_unit).getReferenceImage()),sizeChar);    
        cuMemcpyHtoD(dev_BIN2, Pointer.to(beans.get(admin_unit).getCurrentImage()),  sizeChar);
        cuMemcpyHtoD(dev_ROI,  Pointer.to(beans.get(admin_unit).roi), 				 sizeChar);

    	/*
    	 * 	KERNELS GEOMETRY
    	 * 	NOTE: use ceil() instead of the "%" operator!!!
    	 */
        int bdx, gdx, gdx_2, num_blocks_per_SM, mapel_per_thread;
    	bdx 					= BLOCKDIM;
    	num_blocks_per_SM       = max_threads_per_SM / bdx;
    	mapel_per_thread        = (int)Math.ceil( (double)map_len / (double)((bdx*1)*N_sm*num_blocks_per_SM) );
    	gdx                     = (int)Math.ceil( (double)map_len / (double)( mapel_per_thread*(bdx*1) ) );    	
    	gdx_2					= (int)Math.ceil( (double)map_len / (double)( (bdx*4) ) );
    	
		/*		KERNELS INVOCATION
	   	 *
	   	 *			******************************
	   	 *			-1- imperviousness_change_hist
	   	 *			-2- imperviousness_change
	   	 *			******************************
	   	 *
	   	 *		Note that imperviousness_change_large does not work!!
	   	 */
    	
    	
    	// **HISTC(4)
//		start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_hist4 = Pointer.to(
            Pointer.to(dev_BIN1), Pointer.to(dev_BIN2),
            Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
            Pointer.to(dev_LTAKE_count), Pointer.to(new int[] { mapel_per_thread })
        );
        // Call the kernel function.
        cuLaunchKernel(	F_histc_4,
        				gdx, 1, 1,			// Grid dimension
        				bdx, 1, 1,			// Block dimension
        				0, null,			// Shared memory size and stream
        				kern_hist4, null	// Kernel- and extra parameters
        );
        cuCtxSynchronize();
//		end_t = clock();
//		System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
//		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

        // **CHANGE-MAP
//		start_t = clock();
        // Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
        Pointer kern_chmap = Pointer.to(
            Pointer.to(dev_BIN1),Pointer.to(dev_BIN2),
            Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
            Pointer.to(dev_LTAKE_map)
        );
        // Call the kernel function.
        cuLaunchKernel(	F_chmap,
        				gdx_2, 1, 1,		// Grid dimension
        				bdx*4, 1, 1,		// Block dimension
        				0, null,			// Shared memory size and stream
        				kern_chmap, null		// Kernel- and extra parameters
        );
        cuCtxSynchronize();
/*    	end_t = clock();
    	System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_24,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
    	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/

/*    	System.out.println("______________________________________\n");
    	System.out.println("  %21s\t%6d [msec]\n", "Total time (T):",elapsed_time );
*/

    	//cuMemcpyDtoH(Pointer.to(host_LTAKE_map), 	dev_LTAKE_map, 		sizeInt 	);
    	cuMemcpyDtoH(Pointer.to(host_LTAKE_map), 	dev_LTAKE_map, 		sizeDouble 	);
    	cuMemcpyDtoH(Pointer.to(host_LTAKE_count), 	dev_LTAKE_count, 	sizeInt 	);
    	
    	// CUDA free:
    	cuMemFree( dev_BIN1	);
    	cuMemFree( dev_BIN2	);
    	cuMemFree( dev_LTAKE_map	);
    	cuMemFree( dev_LTAKE_count	);
    	cuMemFree( dev_ROI	);

        // Unload MODULE
        cuModuleUnload(module);

        // Destroy CUDA context:
        cuCtxDestroy(context);

    	//System.out.println("\n CUDA Finished!!\n");
/*		long estimatedTime = System.currentTimeMillis() - startTime;
		System.out.println("Elapsed time fragmentation()" + estimatedTime + " [ms]");
*/
        // conversion(int --> double) & transform(count --> area)
        double outCount[] = new double[4];
        for( int ii=0; ii<4; ii++){
        	outCount[ii] = (double)host_LTAKE_count[ii]*areaOfOnePixel;
        }
        
    	return Arrays.asList(host_LTAKE_map, outCount);
	}// close :: public static double[] land_take

	
	public static double[] potloss_foodsupply(	List<CUDABean> beans,
			double areaOfOnePixel, int admin_unit) {
		/**
		* 	Copyright 2015 Giuliano Langella: ---- testing... ----
		* 	
		* 	This function computes a sort of reduction kernel in driver-API CUDA-C.
		* 	-1- reduction of hist(4)	{ BIN1,BIN2,ROI } --> LTAKE_count 	[kernel::imperviousness_change_histc_sh_4]
		*  
		*  -----------------------------
		*	(N) (BIN2,BIN1)	--> (LTAKE)
		*	-----------------------------
		*	(1) (0,0) 		--> +0			---> nothing changed in rural pixels	LTAKE_count[0]
		*	(2) (0,1) 		--> -1			---> increase of rural pixels			LTAKE_count[1]
		*	(3) (1,0) 		--> +1			---> increase of urban pixels			LTAKE_count[2]
		*	(4) (1,1) 		--> -0			---> nothing changed in urban pixels	LTAKE_count[3]
		*	-----------------------------
		*	where values can be { 0:rural; 1:urban }.
		*/
		
		/*
		* 	NOTES
		* 	??
		*/
		
		/*
		* 	PARAMETERS
		*/
		CUresult err;
		int gpuDeviceCount[]	= { 0 };
		int elapsed_time		= 0;
		int	gpuDev				= 0;
		// count the number of kernels that must print their output:
		int count_print 		= 0;
		
		/*
		* 	ESTABILISH CONTEXT
		*/
		JCudaDriver.setExceptionsEnabled(true);
		// Initialise the driver:
		//CUresult err = 
		cuInit(0);
		
		/*
		*  	RECOGNIZE DEVICE(s) EXISTENCE:
		*/
		// Obtain the number of devices
		cuDeviceGetCount(gpuDeviceCount);
		int deviceCount = gpuDeviceCount[0];
		if (deviceCount == 0) {
		throw new ProcessException("Error: no devices supporting CUDA.");
		}
		/*
		*	SELECT THE FIRST DEVICE
		*	(but I should select/split devices according to streams)
		*/
		int selDev 			= 0;
		CUdevice device 	= new CUdevice();
		cuDeviceGet(device, selDev);
		/*
		* 	QUERY CURRENT GPU PROPERTIES
		*/
		int amountProperty[] 	= { 0 };
		// -1- threads / block
		//cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
		//int maxThreadsPerBlock 	= amountProperty[0];
		// -2- blocks / grid
		//cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
		//int maxGridSize			= amountProperty[0];
		// -3- No. Streaming Multiprocessors
		cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
		int N_sm				= amountProperty[0];
		// -4- threads / SM
		cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
		int max_threads_per_SM	= amountProperty[0];
		
		/*
		* 	CREATE THE CONTEXT
		* 	(for currently selected device)
		*/
		CUcontext context = new CUcontext();
		// int cuCtxCreate_STATUS =
		cuCtxCreate(context, selDev, device);
		// Load the ptx file:
		CUmodule module = new CUmodule();
		cuModuleLoad(module, PTXFILE_land_take);
		/*
		*  ENTRY POINTS
		*  (obtain a function pointer to
		*   the -2- entry functions/kernels)
		*/
		// first kernel
		CUfunction F_histc_4 = new CUfunction();
		cuModuleGetFunction(F_histc_4, module, "_Z32imperviousness_change_histc_sh_4PKhS0_jjPii");
		
		/*
		*  	DIM of ARRAYS in BYTES
		*/
		int WIDTH 		= beans.get(admin_unit).width;
		int HEIGHT		= beans.get(admin_unit).height;
		int map_len 	= WIDTH * HEIGHT;
		long sizeChar	= map_len*Sizeof.BYTE;
		long sizeInt	= map_len*Sizeof.INT;
		long sizeDouble	= map_len*Sizeof.INT;

		/*
		*	CPU ARRAYS
		*/
		int 	host_LTAKE_count[]	= new int[4];
		cuMemAllocHost( 	Pointer.to(host_LTAKE_count),	sizeInt		);
		/*
		* 	GPU ARRAYS
		* 	use CUDA_CHECK_RETURN() for all calls.
		*/    	
		// get pointers
		CUdeviceptr dev_BIN1			= new CUdeviceptr();
		CUdeviceptr dev_BIN2 			= new CUdeviceptr();
		CUdeviceptr dev_ROI 			= new CUdeviceptr();
		CUdeviceptr dev_LTAKE_count 	= new CUdeviceptr();
		// allocate in mem
		cuMemAlloc( dev_BIN1, 			sizeChar	);
		cuMemAlloc( dev_BIN2, 			sizeChar	);
		cuMemAlloc( dev_ROI, 			sizeChar	);
		cuMemAlloc( dev_LTAKE_count,	sizeInt		);
		// set mem
		cudaMemset(	dev_LTAKE_count,0,  4*Sizeof.INT);
		/*
		* 	MEM COPY H2D
		*/
		cuMemcpyHtoD(dev_BIN1, Pointer.to(beans.get(admin_unit).getReferenceImage()),sizeChar);    
		cuMemcpyHtoD(dev_BIN2, Pointer.to(beans.get(admin_unit).getCurrentImage()),  sizeChar);
		cuMemcpyHtoD(dev_ROI,  Pointer.to(beans.get(admin_unit).roi), 				 sizeChar);
		
		/*
		* 	KERNELS GEOMETRY
		* 	NOTE: use ceil() instead of the "%" operator!!!
		*/
		int bdx, gdx, num_blocks_per_SM, mapel_per_thread;
		bdx 					= BLOCKDIM;
		num_blocks_per_SM       = max_threads_per_SM / bdx;
		mapel_per_thread        = (int)Math.ceil( (double)map_len / (double)((bdx*1)*N_sm*num_blocks_per_SM) );
		gdx                     = (int)Math.ceil( (double)map_len / (double)( mapel_per_thread*(bdx*1) ) );    	
		
		/*		KERNELS INVOCATION
		*
		*			******************************
		*			-1- imperviousness_change_hist
		*			******************************
		*/
		
		// **HISTC(4)
		//start_t = clock();
		// Set up the kernel parameters: A pointer to an array of pointers which point to the actual values.
		Pointer kern_hist4 = Pointer.to(
		Pointer.to(dev_BIN1), Pointer.to(dev_BIN2),
		Pointer.to(new int[] { WIDTH }), Pointer.to(new int[] { HEIGHT }),
		Pointer.to(dev_LTAKE_count), Pointer.to(new int[] { mapel_per_thread })
		);
		// Call the kernel function.
		cuLaunchKernel(	F_histc_4,
						gdx, 1, 1,			// Grid dimension
						bdx, 1, 1,			// Block dimension
						0, null,			// Shared memory size and stream
						kern_hist4, null	// Kernel- and extra parameters
		);
		cuCtxSynchronize();
		//end_t = clock();
		//System.out.println("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl_,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
		//elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
				
/*    	System.out.println("______________________________________\n");
		System.out.println("  %21s\t%6d [msec]\n", "Total time (T):",elapsed_time );
		*/
		
		cuMemcpyDtoH(Pointer.to(host_LTAKE_count), 	dev_LTAKE_count, 	sizeInt 	);
		
		// CUDA free:
		cuMemFree( dev_BIN1	);
		cuMemFree( dev_BIN2	);
		cuMemFree( dev_LTAKE_count	);
		
		// Unload MODULE
		cuModuleUnload(module);
		
		// Destroy CUDA context:
		cuCtxDestroy(context);
		
		//System.out.println("\n CUDA Finished!!\n");
		/*		long estimatedTime = System.currentTimeMillis() - startTime;
		System.out.println("Elapsed time fragmentation()" + estimatedTime + " [ms]");
		*/
		// conversion(int --> double) & transform(count --> area)
		double outCount[] = new double[4];
		for( int ii=0; ii<4; ii++){
			outCount[ii] = (double)host_LTAKE_count[ii]*areaOfOnePixel;
		}
		
		return outCount;
	}// close :: public static double[] land_take

}// close :: public class CUDAClass

