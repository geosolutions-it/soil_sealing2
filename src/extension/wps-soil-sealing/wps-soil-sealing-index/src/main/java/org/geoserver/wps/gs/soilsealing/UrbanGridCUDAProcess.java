package org.geoserver.wps.gs.soilsealing;

import it.geosolutions.rendered.viewer.RenderedImageBrowser;

import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.media.jai.DataBufferDouble;
import javax.media.jai.ImageLayout;
import javax.media.jai.JAI;
import javax.media.jai.LookupTableJAI;
import javax.media.jai.PlanarImage;
import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;
import javax.media.jai.operator.LookupDescriptor;
import javax.media.jai.operator.MosaicDescriptor;

import org.geoserver.catalog.FeatureTypeInfo;
import org.geoserver.wps.gs.soilsealing.CLCProcess.StatisticContainer;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridGeometry2D;
import org.geotools.geometry.jts.JTS;
import org.geotools.image.ImageWorker;
import org.geotools.process.ProcessException;
import org.geotools.process.factory.DescribeParameter;
import org.geotools.process.factory.DescribeResult;
import org.geotools.process.gs.GSProcess;
import org.geotools.process.raster.CropCoverage;
import org.geotools.referencing.CRS;
import org.geotools.referencing.operation.transform.AffineTransform2D;
import org.geotools.referencing.operation.transform.ProjectiveTransform;
import org.geotools.resources.image.ImageUtilities;
import org.geotools.xml.xsi.XSISimpleTypes.DateTime;
import org.jaitools.imageutils.ImageLayout2;
import org.jaitools.imageutils.ROIGeometry;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.metadata.spatial.PixelOrientation;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.datum.PixelInCell;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.sun.media.imageioimpl.common.ImageUtil;
import com.vividsolutions.jts.geom.Geometry;

public class UrbanGridCUDAProcess extends UrbanGridProcess implements GSProcess {

    /** Logger used for logging exceptions */
    public static final Logger LOGGER = Logger.getLogger(UrbanGridProcess.class.toString());

    /** Default Pixel Area */
    private static final double PIXEL_AREA = 400;

    private static final CropCoverage CROP = new CropCoverage();

    /** Imperviousness Vectorial Layer */
    private FeatureTypeInfo imperviousnessReference;

    /** Path associated to the shapefile of the reference image */
    private String referenceYear;

    /** Path associated to the shapefile of the current image */
    private String currentYear;

    public UrbanGridCUDAProcess(FeatureTypeInfo imperviousnessReference, String referenceYear,
            String currentYear) {
        super(imperviousnessReference, referenceYear, currentYear);
        this.imperviousnessReference = imperviousnessReference;
        this.referenceYear = referenceYear;
        this.currentYear = currentYear;
    }

    // HP to verify
    // HP1 = admin geometries in Raster space, for index 7a-8-9-10; in SHP CRS for the other indexes
    // HP2 = Coverages already cropped and transformed to the Raster Space

    @DescribeResult(name = "UrbanGridCUDAProcess", description = "Urban Grid indexes calculated using CUDA", type = List.class)
    public List<StatisticContainer> execute(
            @DescribeParameter(name = "reference", description = "Name of the reference raster") GridCoverage2D referenceCoverage,
            @DescribeParameter(name = "now", description = "Name of the new raster") GridCoverage2D nowCoverage,
            @DescribeParameter(name = "index", min = 1, description = "Index to calculate") int index,
            @DescribeParameter(name = "subindex", min = 0, description = "String indicating which sub-index must be calculated") String subId,
            @DescribeParameter(name = "pixelarea", min = 0, description = "Pixel Area") Double pixelArea,
            @DescribeParameter(name = "rois", min = 1, description = "Administrative Areas") List<Geometry> rois,
            @DescribeParameter(name = "populations", min = 0, description = "Populations for each Area") List<List<Integer>> populations,
            @DescribeParameter(name = "coefficient", min = 0, description = "Multiplier coefficient for index 10") Double coeff) {

        // Check on the index 7
        boolean nullSubId = subId == null || subId.isEmpty();
        boolean subIndexA = !nullSubId && subId.equalsIgnoreCase("a");
        boolean subIndexC = !nullSubId && subId.equalsIgnoreCase("c");
        boolean subIndexB = !nullSubId && subId.equalsIgnoreCase("b");
        if (index == SEVENTH_INDEX && (nullSubId || !(subIndexA || subIndexB || subIndexC))) {
            throw new IllegalArgumentException("Wrong subindex for index 7");
        }
        // Check if almost one coverage is present
        if (referenceCoverage == null && nowCoverage == null) {
            throw new IllegalArgumentException("No input Coverage provided");
        }

        double areaPx;
        if (pixelArea == null) {
            areaPx = PIXEL_AREA;
        } else {
            areaPx = pixelArea;
        }

        // Check if Geometry area or perimeter must be calculated
        boolean inRasterSpace = true;
        // Selection of the operation to do for each index
        switch (index) {
        case FIFTH_INDEX:
        case SIXTH_INDEX:
        case SEVENTH_INDEX:
            if (!subIndexA) {
                inRasterSpace = false;
            }
            break;
        default:
            break;
        }

        // If the index is 7a-8-9-10 then the input Geometries must be transformed to the Model Space
        List<Geometry> geoms = new ArrayList<Geometry>();
        final AffineTransform gridToWorldCorner = (AffineTransform) ((GridGeometry2D) referenceCoverage
                .getGridGeometry()).getGridToCRS2D(PixelOrientation.UPPER_LEFT);
        if (inRasterSpace) {
            for (Geometry geo : rois) {
                try {
                    geoms.add(JTS.transform(geo, ProjectiveTransform.create(gridToWorldCorner)));
                } catch (MismatchedDimensionException e) {
                    LOGGER.log(Level.SEVERE, e.getMessage(), e);
                    throw new ProcessException(e);
                } catch (TransformException e) {
                    LOGGER.log(Level.SEVERE, e.getMessage(), e);
                    throw new ProcessException(e);
                }
            }
        } else {
            geoms.addAll(rois);
        }

        // Check if the Geometries must be reprojected
/*        Object userData = geoms.get(0).getUserData();
        if (!inRasterSpace && userData instanceof CoordinateReferenceSystem) {
            CoordinateReferenceSystem geomCRS = (CoordinateReferenceSystem) userData;
            CoordinateReferenceSystem refCRS = referenceCoverage.getCoordinateReferenceSystem();
            MathTransform tr = null;
            try {
                tr = CRS.findMathTransform(geomCRS, refCRS);

                if (!(tr == null || tr.isIdentity())) {
                    int geosize = geoms.size();
                    for (int i = 0; i < geosize; i++) {
                        Geometry geo = geoms.get(i);
                        Geometry transform = JTS.transform(geo, tr);
                        transform.setUserData(refCRS);
                        geoms.set(i, transform);
                    }
                }
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, e.getMessage(), e);
                throw new ProcessException(e);
            }
            // Otherwise only set the correct User_Data parameter
        } else if (inRasterSpace){
*/          int geosize = geoms.size();
            for (int i = 0; i < geosize; i++) {
                Geometry geo = geoms.get(i);
                geo.setUserData(referenceCoverage.getCoordinateReferenceSystem());
            }
//        }

        // Empty arrays containing the statistics results
        double[] statsRef = null;
        double[] statsNow = null;

        // Create a new List of CUDA Bean objects
        List<CUDABean> beans = new ArrayList<CUDABean>();

        // Loop around all the Geometries and generate a new CUDA Bean object
        try {
            //MathTransform transform = ProjectiveTransform.create(gridToWorldCorner).inverse();
            int counter = 0;
            for (Geometry geo : geoms) {
                // Create the CUDABean object
                CUDABean bean = new CUDABean();

                // Populate it with Reference coverage parameters
                populateBean(bean, true, referenceCoverage, geo, null);

                // Set the population values if needed
                if (populations != null) {
                    Integer popRef = populations.get(0).get(counter);
                    bean.setPopRef(popRef);
                }

                // Do the same for the Current Coverage if present
                if (nowCoverage != null) {
                    populateBean(bean, false, nowCoverage, geo, null);
                    // Set the population values if needed
                    if (populations != null) {
                        Integer popCur = populations.get(1).get(counter);
                        bean.setPopCur(popCur);
                    }
                }
                // Add the bean to the list
                beans.add(bean);
                // Update counter
                counter++;
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new ProcessException(e);
        }

        // Calculate the index using CUDA
//		System.out.println( java.text.DateFormat.getDateTimeInstance().format(Calendar.getInstance().getTime()) );
//      long startTime = System.currentTimeMillis();
        /**
         * 	Generalize:
         * 		> isUrban = false/true			------------------------|
         * 		> RADIUS [meters] = scalar		---------------------------------|
         */
        boolean isRural = false;// link to GUI
        int RADIUS 		= 100;// [m] // link to GUI
        Object output = calculateCUDAIndex(index, subId, areaPx, beans, isRural, RADIUS);
//		long estimatedTime = System.currentTimeMillis() - startTime;
//		System.out.println("Elapsed time calculateCUDAIndex()\t--> " + estimatedTime + " [ms]");
		Rectangle refRect = PlanarImage.wrapRenderedImage(referenceCoverage.getRenderedImage()).getBounds();
        
		// URBAN DIFFUSION
        if (index==7){
        	// do new CUDA processing!
        }
        // For index 8 calculate the final Image
        else if (index == 8) {
        	System.out.println("rural="+isRural + " -- radius="+RADIUS+" [m]");
            List<StatisticContainer> results = new ArrayList<CLCProcess.StatisticContainer>();
            StatisticContainer stats = new StatisticContainer();
            double[][][] images = (double[][][]) output;
            double[][] refData = images[0];
            int numGeo = beans.size();
            // Images to mosaic
            RenderedImage[] refImgs = new RenderedImage[numGeo];
            ROI[] roiObjs = new ROI[numGeo];

            // Giuliano tested for 91 municipalities in NAPLES and it FAILED within the following FOR loop!!
            for (int i = 0; i < numGeo; i++) {
                CUDABean bean = beans.get(i);
                double[] data = refData[i];
                Rectangle rect = new Rectangle(bean.getMinX(), bean.getMinY(), bean.getWidth(),
                        bean.getHeight());
                refImgs[i] = createImage(rect, data);
                roiObjs[i] = bean.getRoiObj();
            }
            ImageLayout layout = new ImageLayout2();
            layout.setMinX(refRect.x);
            layout.setMinY(refRect.y);
            layout.setWidth(refRect.width);
            layout.setHeight(refRect.height);

            RenderingHints hints = new RenderingHints(JAI.KEY_IMAGE_LAYOUT, layout);
            
            // Mosaic of the images
            double[] background = new double[]{-1.0};
			RenderedImage finalRef = MosaicDescriptor.create(refImgs,
                    MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null, roiObjs, null,
                    background, hints);

             //RenderedImageBrowser.showChain(finalRef, false, false);

            // Upgrade of the statistics container
            stats.setReferenceImage(finalRef);
            // Check if the same calculations must be done for the Current coverage
            if (nowCoverage != null) {
                double[][] curData = images[1];
                double[][] diffData = images[2];
                RenderedImage[] currImgs = new RenderedImage[numGeo];
                RenderedImage[] diffImgs = new RenderedImage[numGeo];
                for (int i = 0; i < numGeo; i++) {
                    CUDABean bean = beans.get(i);
                    double[] data = curData[i];
                    double[] diff = diffData[i];
                    Rectangle rect = new Rectangle(bean.getMinX(), bean.getMinY(), bean.getWidth(),
                            bean.getHeight());
                    currImgs[i] = createImage(rect, data);
                    diffImgs[i] = createImage(rect, diff);
                }
                // Mosaic of the images
                RenderedImage finalCurr = MosaicDescriptor.create(currImgs,
                        MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null,
                        roiObjs, null, background, hints);

                // Mosaic of the images
                RenderedImage finalDiff = MosaicDescriptor.create(diffImgs,
                        MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null,
                        roiObjs, null, background, hints);
                // Update the statistics container
                stats.setNowImage(finalCurr);
                stats.setDiffImage(finalDiff);
            }
            results.add(stats);
            return results;
        }
        else if (index == 9) {// LAND TAKE
            double[][] values = (double[][]) output;
            statsRef = values[0];
            statsNow = values.length > 1 ? values[1] : null;
        }
        else {
            double[][] values = (double[][]) output;
            statsRef = values[0];
            statsNow = values.length > 1 ? values[1] : null;
        }

        // Result accumulation
        List<StatisticContainer> results = accumulateResults(rois, statsRef, statsNow);

        return results;

    }

    /**
     * Method for calculating the selected index using the underlying CUDA
     * 
     * @param index
     * @param subId
     * @param areaPx
     * @param statsRef
     * @param statsNow
     * @param beans
     */
    private Object calculateCUDAIndex(int index, String subId, double areaPx, List<CUDABean> beans,
            boolean rural, int ray_meters) {
        /*	NOTES:
         * 		(1)	I don't like numeric index, because we might change order (or whatever).
         * 			A string index might be more general.
         * 		(2)	We need to define how to pass the correct year when 1 year (ref or curr)
         * 			is selected from within the MapStore GUI.
         * 			I think that one solution could be to delete the for loop on n_years, 
         * 			and call once fragmentation code using one of 1/2/3 (1:ref; 2:curr; 3:diff)
         * 			(when 3:diff is passed, JCuda has to launch the set of kernels 3 times, so it
         * 			
         * 		(3) We have to pass in input the sub-index value, in order to compute the correct
         * 			7th index (and delete subId_tmp accordingly).
         */

        // FOR INDEX 8 I WOULD EXPECT A MATRIX 1/3 x N_GEOMETRIES X (IMAGE DATA)
        // WHERE
    	// 	LINE 0 == REFERENCE TIME DATA
        // 	LINE 1 == CURRENT TIME DATA
        // 	LINE 2 == DIFFERENCE BETWEEN THE OTHER DATA
        // MATRIX DATA TYPE SHOULD BE BYTE
    	
    	/**	Change as required
    	 */
    	// number of administrative units:
    	int n_adm_units 	= beans.size();
        int n_years 		= beans.get(0).getCurrentImage() != null ? 2 : 1;
    	int CELLSIZE 		= (int)Math.sqrt(areaPx);//beans.get(0).
    	int ray_pixels 		= ray_meters/CELLSIZE;
        double[][][] result = new double[n_adm_units][3][];
    	
        // I WOULD EXPECT A 2 X N_GEOMETRIS MATRIX
        // WHERE LINE 0 == INDEX VALUES FOR REFERENCE TIME
        // WHERE LINE 1 == INDEX VALUES FOR CURRENT TIME
        //double[][] result = new double[n_adm_units][n_years];
        //if (n_years == 2) {
        //	some cuda code
    	//}
        	switch( index ){

        		// ------- CORINE LAND COVER -------
        		case 1:  // coverage coefficient
        			System.out.println("Not yet implemented");
        			return null;
        			
        		case 2:  // rate of change
        			System.out.println("Not yet implemented");
        			return null;
        			
        		case 3:  // marginal land take
        			System.out.println("Not yet implemented");
        			return null;
        			
        		case 4:  // urban sprawl
        			System.out.println("Not yet implemented");
        			return null;
        		
        		// ------- IMPERVIOUSNESS ------- 	
        		case 5:  // urban sprawl [ccl+sum(hist(~polyMax))]
        			//result[i] = CUDAClass.urban_sprawl( beans, rural, areaPx, rayIndex );
        			return null;
        		case 6:  // edge density [perimeter]
        			//result[i] = CUDAClass.edge_density( beans, rural, areaPx, rayIndex );
        			return null;
        		case 7:  // urban diffusion
        			if(subId.equalsIgnoreCase("a")){ 		// urban_area [reduction]* --> a modification from original occurred!
        				//result[i] = CUDAClass.urban_area( beans, rural, areaPx, rayIndex );
        			}else if(subId.equalsIgnoreCase("b")){	// area of polygon with maximum extension [ccl+ave(hist(~polyMax))]
        				//result[i] = CUDAClass.Pmax_area( beans, rural, areaPx, rayIndex );
        			}else if(subId.equalsIgnoreCase("c")){	// average area of other polygons [ccl+hist]
        				//result[i] = CUDAClass.Pother_Marea( beans, rural, areaPx, rayIndex );
        			}
        			return null;
        		case 8:  // FRAGMENTATION [myFragProg]
                    //double[][][] result = new double[n_adm_units][3][];
        			/*
        			 * result = [admin(i)] [year=3] [maplen=HEIGHT(i)*WIDTH(i)]
        			 */
                    // loop for each administrative unit
                    for (int j = 0; j < n_adm_units; j++) {	// lauch for administrative units
                    	// number of grids per admin_unit to give in output:
                    	for (int i = 0; i < n_years; i++) {	// launch for ref/curr/diff
                    		result[j][i]  = CUDAClass.fragmentation( beans, rural, areaPx, ray_pixels, i, j );
                        }
                    	if (n_years>1){
                    		result[j][2] = result[j][1];
        /*                	for(int ii=0;ii<beans.get(j).width*beans.get(j).height;ii++){
                        		result[j][n_years][ii] = result[j][1][ii]-result[j][0][ii];
                        	}*/
                    	}
        			}
                    return result;
        		case 9:	 // LAND TAKE
        			/*	IMPORTANT:
        			 * 		1.\ I have to add ROI in computation 
        			 * result = [admin(i)] [0] [maplen=HEIGHT(i)*WIDTH(i)]
        			 * 			[admin(i)] [1] [Counts=4]
        			 */
                    // loop for each administrative unit
                    for (int j = 0; j < n_adm_units; j++) {
                    	final List<double[]> resultCuda = CUDAClass.land_take( beans, areaPx, j );
                    	// rearrange to fit the middleware process and GUI code requirements----
                    	result[j][0] = resultCuda.get(0);// MAP
                    	result[j][1] = resultCuda.get(1);// COUNTS "+1" (in m2) --> transform in [ha]
                    	result[j][2] = resultCuda.get(1);// COUNTS "-1" (in m2) --> transform in [ha]
                    	//---------
                    }
        			return result;
        		case 10: // potential loss of food supply
        			/*
        			 * result = [admin(i)] [0] [maplen=HEIGHT(i)*WIDTH(i)]
        			 * 			[admin(i)] [1] [Counts=4]
        			 */
        			//result[i] = CUDAClass.potloss_foodsupply( beans, rural, areaPx, rayIndex );
                    for (int j = 0; j < n_adm_units; j++) {
                    	// rearrange to fit the middleware process and GUI code requirements----
                    	// COUNTS (in m2) --> transform in [ha]
                    	result[j][0] = CUDAClass.potloss_foodsupply(beans, areaPx, j);
                    	//---------
                    }
        			return result;
        	}
		return null;
    }

    /**
     * Quick method for populating the {@link CUDABean} instance provided.
     * 
     * @param bean
     * @param reference
     * @param coverage
     * @param geo
     * @param transform
     * @throws IOException
     * @throws MismatchedDimensionException
     * @throws TransformException
     */
    private void populateBean(CUDABean bean, boolean reference, GridCoverage2D coverage,
            Geometry geo, MathTransform transform) throws IOException,
            MismatchedDimensionException, TransformException {
        // 1) Crop the two coverages with the selected Geometry
        GridCoverage2D crop = CROP.execute(coverage, geo, null);
        transform = ProjectiveTransform.create((AffineTransform)crop.getGridGeometry().getGridToCRS(PixelInCell.CELL_CORNER)).inverse();
        // 2) Extract the BufferedImage from each image
        RenderedImage image = crop.getRenderedImage();
        Rectangle rectIMG = new Rectangle(image.getMinX(), image.getMinY(), image.getWidth(),
                image.getHeight());
        ImageWorker w = new ImageWorker(image);
        BufferedImage buf = w.getBufferedImage();
        if (image instanceof RenderedOp) {
            ((RenderedOp) image).dispose();
        }

        // 3) Generate an array of data from each image
        Raster data = buf.getData();
        final DataBufferByte db = (DataBufferByte) data.getDataBuffer();
        byte[] byteData = db.getData();

        if (reference) {
            // 4) Transform the Geometry to Raster space
            Geometry rs = JTS.transform(geo, transform);
            ROI roiGeo = new ROIGeometry(rs);

            // 5) Extract an array of data from the transformed ROI
            byte[] roiData = getROIData(roiGeo, rectIMG);
            bean.setRoi(roiData);
            bean.setRoiObj(roiGeo);

            // 6) Setting the Coverage data array
            bean.setReferenceImage(byteData);

            // 7) Setting the Image dimensions
            bean.setHeight(rectIMG.height);
            bean.setWidth(rectIMG.width);
            bean.setMinX(rectIMG.x);
            bean.setMinY(rectIMG.y);
        } else {
            // 6) Setting the Coverage data array
            bean.setCurrentImage(byteData);
        }
    }

    private byte[] getROIData(ROI roi, Rectangle rectIMG) {
        byte[] dataROI;
        PlanarImage roiIMG = roi.getAsImage();
        Rectangle rectROI = roiIMG.getBounds();
        // Forcing to component colormodel in order to avoid packed bits
        ImageWorker w = new ImageWorker();
        w.setImage(roiIMG);
        w.forceComponentColorModel();
        RenderedImage img = w.getRenderedImage();
        //
        BufferedImage test = new BufferedImage(rectIMG.width, rectIMG.height,
                BufferedImage.TYPE_BYTE_GRAY);
        ImageLayout2 layout = new ImageLayout2(test);
        layout.setMinX(img.getMinX());
        layout.setMinY(img.getMinY());
        layout.setWidth(img.getWidth());
        layout.setHeight(img.getHeight());
        // Lookup
        byte[] lut = new byte[256];
        lut[255] = 1;
        lut[1] = 1;
        LookupTableJAI table = new LookupTableJAI(lut);
        RenderingHints hints = new RenderingHints(JAI.KEY_IMAGE_LAYOUT, layout);
        RenderedOp transformed = LookupDescriptor.create(img, table, hints);

        Graphics2D gc2d = null;
        // Translation parameters in order to position the ROI data correctly in the Raster Space
        int trX = -rectIMG.x + rectROI.x - rectIMG.x;
        int trY = -rectIMG.y + rectROI.y - rectIMG.y;
        try {
            gc2d = test.createGraphics();
            gc2d.drawRenderedImage(transformed, AffineTransform.getTranslateInstance(trX, trY));
        } finally {
            gc2d.dispose();
        }
        Rectangle testRect = new Rectangle(rectIMG.width, rectIMG.height);
        DataBufferByte dbRoi = (DataBufferByte) test.getData(testRect).getDataBuffer();
        dataROI = dbRoi.getData();
        // BufferedImage is stored in memory so the planarImage chain before can be disposed
        ImageUtilities.disposePlanarImageChain(transformed);
        // Flush of the BufferedImage
        test.flush();

        return dataROI;
    }

    /**
     * Creates an image from an array of {@link Integer}
     * 
     * @param rect
     * @param data
     * @return
     */
    private RenderedImage createImage(Rectangle rect, final double[] data) {
        // Definition of the SampleModel
        final SampleModel sm = new PixelInterleavedSampleModel(DataBuffer.TYPE_DOUBLE, rect.width,
                rect.height, 1, rect.width, new int[] { 0 });
        // DataBuffer containing input data
        final DataBufferDouble db1 = new DataBufferDouble(data, rect.width * rect.height);
        // Writable Raster used for creating the BufferedImage
        final WritableRaster wr = com.sun.media.jai.codecimpl.util.RasterFactory
                .createWritableRaster(sm, db1, new Point(0, 0));
        final BufferedImage image = new BufferedImage(ImageUtil.createColorModel(sm), wr, false,
                null);

        ImageWorker w = new ImageWorker(image);
        w.tile();
        if (rect.x != 0 || rect.y != 0) {
            w.affine(AffineTransform.getTranslateInstance(rect.x, rect.y), null, null);
        }
        return w.getRenderedImage();
    }

    /**
     * Static class containing all the data for a single Geometry instance which will be passed to CUDA
     * 
     * @author geosolutions
     * 
     */
    static class CUDABean {
        /** Reference Coverage data array */
        byte[] referenceImage;

        /** Current Coverage data array */
        private byte[] currentImage;

        /** ROI data array */
        byte[] roi;

        /** Image Width */
        int width;

        /** Image Height */
        int height;

        /** Reference value for population */
        private Integer popRef;

        /** Current value for population */
        private Integer popCur;

        /** Image minX value */
        private int minX;

        /** Image minY value */
        private int minY;
        
        /** ROI value */
        private ROI roiObj;

        public byte[] getReferenceImage() {
            return referenceImage;
        }

        public void setMinX(int minX) {
            this.minX = minX;
        }

        public void setMinY(int minY) {
            this.minY = minY;
        }

        public void setReferenceImage(byte[] referenceImage) {
            this.referenceImage = referenceImage;
        }

        public byte[] getCurrentImage() {
            return currentImage;
        }

        public void setCurrentImage(byte[] currentImage) {
            this.currentImage = currentImage;
        }

        public byte[] getRoi() {
            return roi;
        }

        public void setRoi(byte[] roi) {
            this.roi = roi;
        }

        public int getWidth() {
            return width;
        }

        public void setWidth(int width) {
            this.width = width;
        }

        public int getHeight() {
            return height;
        }

        public void setHeight(int height) {
            this.height = height;
        }

        public Integer getPopRef() {
            return popRef;
        }

        public void setPopRef(Integer popRef) {
            this.popRef = popRef;
        }

        public Integer getPopCur() {
            return popCur;
        }

        public void setPopCur(Integer popCur) {
            this.popCur = popCur;
        }

        public int getMinX() {
            return minX;
        }

        public int getMinY() {
            return minY;
        }

		public ROI getRoiObj() {
			return roiObj;
		}

		public void setRoiObj(ROI roiObj) {
			this.roiObj = roiObj;
		}
    }
}
