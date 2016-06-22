package org.geoserver.wps.gs.soilsealing;

import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
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
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingSubIndexType;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridGeometry2D;
import org.geotools.geometry.Envelope2D;
import org.geotools.geometry.GeneralEnvelope;
import org.geotools.geometry.jts.JTS;
import org.geotools.image.ImageWorker;
import org.geotools.process.ProcessException;
import org.geotools.process.factory.DescribeParameter;
import org.geotools.process.factory.DescribeResult;
import org.geotools.process.gs.GSProcess;
import org.geotools.process.raster.CropCoverage;
import org.geotools.referencing.CRS;
import org.geotools.referencing.operation.transform.ProjectiveTransform;
import org.geotools.resources.image.ImageUtilities;
import org.jaitools.imageutils.ImageLayout2;
import org.jaitools.imageutils.ROIGeometry;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.metadata.spatial.PixelOrientation;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.datum.PixelInCell;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.sun.media.imageioimpl.common.ImageUtil;
import com.vividsolutions.jts.geom.Geometry;

@SuppressWarnings("deprecation")
public class UrbanGridCUDAProcess extends UrbanGridProcess implements GSProcess {

    /** Logger used for logging exceptions */
    public static final Logger LOGGER = Logger.getLogger(UrbanGridProcess.class.toString());

    /** Select Current Cuda Device */
    // public static final int gpuDevice = 0;

    /** Default Pixel Area */
    private static final double PIXEL_AREA = 400;
    
    /** Default NODATAVALUE */
    // ––> to be used in place of zero when it conflicts with map results (e.g. fragmentation)
    //private static final double NODATA_VALUE = -99;
    
    private static final CropCoverage CROP = new CropCoverage();

    /** Imperviousness Vectorial Layer */
    @SuppressWarnings("unused")
    private FeatureTypeInfo imperviousnessReference;

    /** Path associated to the shapefile of the reference image */
    @SuppressWarnings("unused")
    private String referenceYear;

    /** Path associated to the shapefile of the current image */
    @SuppressWarnings("unused")
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
            @DescribeParameter(name = "index", min = 1, description = "Index to calculate") SoilSealingIndexType soilSealingIndexType,
            @DescribeParameter(name = "subindex", min = 0, description = "String indicating which sub-index must be calculated") SoilSealingSubIndexType subId,
            @DescribeParameter(name = "pixelarea", min = 0, description = "Pixel Area") Double pixelArea,
            @DescribeParameter(name = "rois", min = 1, description = "Administrative Areas") List<Geometry> rois,
            @DescribeParameter(name = "populations", min = 0, description = "Populations for each Area") List<List<Integer>> populations,
            @DescribeParameter(name = "coefficient", min = 0, description = "Multiplier coefficient for index 10") Double coeff,
            @DescribeParameter(name = "rural", min = 0, description = "Rural or Urban index") boolean rural,
            @DescribeParameter(name = "radius", min = 0, description = "Radius in meters") int radius,
            @DescribeParameter(name = "waterBodiesMaskReference", min = 0, description = "Name of the water bodies mask layer, optionally fully qualified (workspace:name)") FeatureTypeInfo waterBodiesMaskReference)
   
            throws IOException {

        // Checks on the index 7 "Dispersive Urban Growth"
        boolean nullSubId = subId == null;
        boolean subIndexA = !nullSubId && subId == SoilSealingSubIndexType.URBAN_AREA;
        boolean subIndexC = !nullSubId && subId == SoilSealingSubIndexType.HIGHEST_POLYGON_RATIO;
        boolean subIndexB = !nullSubId && subId == SoilSealingSubIndexType.OTHER_POLYGONS_RATIO;

        if (soilSealingIndexType == SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH
                && (nullSubId || !(subIndexA || subIndexB || subIndexC))) {
            throw new IllegalArgumentException(
                    "Wrong subindex for index 7 [" + soilSealingIndexType + "]");
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
        switch (soilSealingIndexType) {
        case URBAN_DISPERSION:
        case EDGE_DENSITY:
        case DISPERSIVE_URBAN_GROWTH:
        case FRAGMENTATION:
        case MODEL_URBAN_DEVELOPMENT:
        case NEW_URBANIZATION:
        case NEW_ECO_CORRIDOR:
            //if (!subIndexA) {
                inRasterSpace = false;
            //}
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
        /*
         * Object userData = geoms.get(0).getUserData(); if (!inRasterSpace && userData instanceof CoordinateReferenceSystem) {
         * CoordinateReferenceSystem geomCRS = (CoordinateReferenceSystem) userData; CoordinateReferenceSystem refCRS =
         * referenceCoverage.getCoordinateReferenceSystem(); MathTransform tr = null; try { tr = CRS.findMathTransform(geomCRS, refCRS);
         * 
         * if (!(tr == null || tr.isIdentity())) { int geosize = geoms.size(); for (int i = 0; i < geosize; i++) { Geometry geo = geoms.get(i);
         * Geometry transform = JTS.transform(geo, tr); transform.setUserData(refCRS); geoms.set(i, transform); } } } catch (Exception e) {
         * LOGGER.log(Level.SEVERE, e.getMessage(), e); throw new ProcessException(e); } // Otherwise only set the correct User_Data parameter } else
         * if (inRasterSpace){
         */ 
        int geosize = geoms.size();
        final CoordinateReferenceSystem refCrs = referenceCoverage.getCoordinateReferenceSystem();
        for (int i = 0; i < geosize; i++) {
            Geometry geo = geoms.get(i);

            geo.setUserData(refCrs);

            if (geo.getSRID() == 0) {
                try {
                    geo.setSRID(CRS.lookupEpsgCode(refCrs, true));
                } catch (FactoryException e) {
                    LOGGER.log(Level.WARNING, e.getMessage(), e);
                }
            }
        }
        // }

        // Empty arrays containing the statistics results
        double[] statsRef = null;
        double[] statsNow = null;
        double[][][] statsComplex = null;

        // Create a new List of CUDA Bean objects
        List<CUDABean> beans = new ArrayList<CUDABean>();

        // Loop around all the Geometries and generate a new CUDA Bean object
        try {
            // MathTransform transform = ProjectiveTransform.create(gridToWorldCorner).inverse();
            int counter = 0;
            int buffer = (
                    soilSealingIndexType == SoilSealingIndexType.NEW_URBANIZATION || 
                    soilSealingIndexType == SoilSealingIndexType.NEW_ECO_CORRIDOR ? radius : 0);
            for (Geometry geo : geoms) {
                // Create the CUDABean object
                CUDABean bean = new CUDABean();
                bean.setAreaPix(areaPx);

                // Populate it with Reference coverage parameters
                try {
                    populateBean(bean, true, referenceCoverage, geo, null, buffer, soilSealingIndexType, waterBodiesMaskReference);

                    // Set the population values if needed
                    if (populations != null) {
                        Integer popRef = populations.get(0).get(counter);
                        bean.setPopRef(popRef);
                    }
    
                    // Do the same for the Current Coverage if present
                    if (nowCoverage != null) {
                        populateBean(bean, false, nowCoverage, geo, null, buffer, soilSealingIndexType, waterBodiesMaskReference);
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
                } catch (Exception e) {
                    LOGGER.log(Level.WARNING, geo.toText(), e);
                }
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new ProcessException(e);
        }

        // Calculate the index using CUDA
        // System.out.println( java.text.DateFormat.getDateTimeInstance().format(Calendar.getInstance().getTime()) );
        // long startTime = System.currentTimeMillis();
        /**
         * Generalize:
         *  > isUrban = false/true              ------------------------|
         *  > RADIUS [meters] = scalar ---------------------------------|
         */
        Object output = null;
        try {
            output = calculateCUDAIndex(soilSealingIndexType, subId, beans, rural, radius);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        // long estimatedTime = System.currentTimeMillis() - startTime;
        // System.out.println("Elapsed time calculateCUDAIndex()\t--> " + estimatedTime + " [ms]");
        Rectangle refRect = PlanarImage.wrapRenderedImage(referenceCoverage.getRenderedImage())
                .getBounds();

        // For index 8 calculate the final Image
        if (soilSealingIndexType == SoilSealingIndexType.FRAGMENTATION
                || soilSealingIndexType == SoilSealingIndexType.LAND_TAKE
                || soilSealingIndexType == SoilSealingIndexType.NEW_URBANIZATION
                || soilSealingIndexType == SoilSealingIndexType.NEW_ECO_CORRIDOR) {

            LOGGER.fine("rural=" + rural + " -- radius/buffer=" + radius + " [m]");

            List<StatisticContainer> results = new ArrayList<CLCProcess.StatisticContainer>();
            StatisticContainer stats = new StatisticContainer();
            double[][][] images = (double[][][]) output;

            int numGeo = beans.size();
            // Images to mosaic
            RenderedImage[] refImgs = new RenderedImage[numGeo];
            ROI[] roiObjs = new ROI[numGeo];

            // Giuliano tested for 91 municipalities in NAPLES and it FAILED within the following FOR loop!!
            for (int i = 0; i < numGeo; i++) {
                double[][] refData = images[i];
                CUDABean bean = beans.get(i);
                double[] data = refData[0];
                if (data != null) {
                    Rectangle rect = new Rectangle(bean.getMinX(), bean.getMinY(), bean.getWidth(),
                            bean.getHeight());
                    refImgs[i] = createImage(rect, data);
                    roiObjs[i] = bean.getRoiObj();
                }
            }
            ImageLayout layout = new ImageLayout2();
            layout.setMinX(refRect.x);
            layout.setMinY(refRect.y);
            layout.setWidth(refRect.width);
            layout.setHeight(refRect.height);

            RenderingHints hints = new RenderingHints(JAI.KEY_IMAGE_LAYOUT, layout);

            // Mosaic of the images
            double[] background = (
                    soilSealingIndexType == SoilSealingIndexType.FRAGMENTATION || 
                    soilSealingIndexType == SoilSealingIndexType.NEW_URBANIZATION ||
                    soilSealingIndexType == SoilSealingIndexType.NEW_ECO_CORRIDOR
                            ? new double[] { SoilSealingImperviousnessProcess.FRAG_NODATA } : 
                                new double[] { SoilSealingImperviousnessProcess.NODATA });
            RenderedImage finalRef = MosaicDescriptor.create(refImgs,
                    MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null, roiObjs, null, background, hints);

            // RenderedImageBrowser.showChain(finalRef, false, false);

            // Upgrade of the statistics container
            stats.setReferenceImage(finalRef);
            // Check if the same calculations must be done for the Current coverage
            if (nowCoverage != null && soilSealingIndexType != SoilSealingIndexType.LAND_TAKE) {
                RenderedImage[] currImgs = new RenderedImage[numGeo];
                RenderedImage[] diffImgs = new RenderedImage[numGeo];
                for (int i = 0; i < numGeo; i++) {
                    CUDABean bean = beans.get(i);
                    double[] data = images[i][1];
                    double[] diff = images[i][2];
                    Rectangle rect = new Rectangle(bean.getMinX(), bean.getMinY(), bean.getWidth(),
                            bean.getHeight());
                    currImgs[i] = createImage(rect, data);
                    diffImgs[i] = createImage(rect, diff);
                }
                // Mosaic of the images
                RenderedImage finalCurr = MosaicDescriptor.create(currImgs,
                        MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null, roiObjs, null, background,
                        hints);

                // Mosaic of the images
                RenderedImage finalDiff = MosaicDescriptor.create(diffImgs,
                        MosaicDescriptor.MOSAIC_TYPE_OVERLAY, null, roiObjs, null, background,
                        hints);
                // Update the statistics container
                stats.setNowImage(finalCurr);
                stats.setDiffImage(finalDiff);
            }
            results.add(stats);
            return results;
        }
        /*
         * else if (index == 9) {// LAND TAKE double[][][] values = (double[][][]) output; statsRef = values[0][0]; statsNow = values[0].length > 1 ?
         * values[0][1] : null; }
         */
        else if (soilSealingIndexType == SoilSealingIndexType.MODEL_URBAN_DEVELOPMENT) {
            statsComplex = (double[][][]) output;
        } else {
            double[][][] values = (double[][][]) output;
            statsRef = new double[values.length];
            statsNow = (values[0].length > 1 && values[0][1] != null? new double[values.length] : null);

            for (int v = 0; v < values.length; v++) {
                statsRef[v] = values[v][0][0];
                if (statsNow != null) {
                    statsNow[v] = values[v][1][0];
                }
            }
        }

        // Result accumulation
        List<StatisticContainer> results = accumulateResults(rois, statsRef, statsNow,
                statsComplex);

        return results;

    }

    /**
     * Method for calculating the selected index using the underlying CUDA
     * 
     * @param soilSealingIndexType
     * @param subId
     * @param areaPx
     * @param statsRef
     * @param statsNow
     * @param beans
     * @throws IOException
     */
    @SuppressWarnings("static-access")
    private Object calculateCUDAIndex(SoilSealingIndexType soilSealingIndexType,
            SoilSealingSubIndexType subId, List<CUDABean> beans, boolean rural, int ray_meters)
            throws IOException {
        /*
         * NOTES: (1) I don't like numeric index, because we might change order (or whatever). A string index might be more general. (2) We need to
         * define how to pass the correct year when 1 year (ref or curr) is selected from within the MapStore GUI. I think that one solution could be
         * to delete the for loop on n_years, and call once fragmentation code using one of 1/2/3 (1:ref; 2:curr; 3:diff) (when 3:diff is passed,
         * JCuda has to launch the set of kernels 3 times, so it
         * 
         * (3) We have to pass in input the sub-index value, in order to compute the correct 7th index (and delete subId_tmp accordingly).
         */

        // FOR INDEX 8 I WOULD EXPECT A MATRIX 1/3 x N_GEOMETRIES X (IMAGE DATA)
        // WHERE
        // LINE 0 == REFERENCE TIME DATA
        // LINE 1 == CURRENT TIME DATA
        // LINE 2 == DIFFERENCE BETWEEN THE OTHER DATA
        // MATRIX DATA TYPE SHOULD BE BYTE

        /**
         * Change as required
         */
        // number of administrative units:
        int n_adm_units = beans.size();
        int n_years = beans.get(0).getCurrentImage() != null ? 2 : 1;
        double areaPix = beans.get(0).getAreaPx();
        int CELLSIZE = beans.get(0).getCellSize();
        int ray_pixels = ray_meters / CELLSIZE;
        double[][][] result = new double[n_adm_units][3][];
        int Distribution = 90;// P90! but we should put this on the GUI...

        // ***IMPERVIOUSENESS STACK :: save BIN & ROI on hdd***
        // I assume that the test is performed using ONE admin and ONE year.
        // I should add other Input parameters according to the selected index.
        // I save other intermediate results on hdd within the corresponding java class
        if (SoilSealingTestUtils.TESTING && soilSealingIndexType.getIdx() > 4) {
            try {
                int WIDTH = beans.get(0).width;
                int HEIGHT = beans.get(0).height;
                // ROI
                SoilSealingTestUtils.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT, beans.get(0).roi,
                        DataBuffer.TYPE_BYTE, "ssgci_roi");
                SoilSealingTestUtils.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT, beans.get(0).getReferenceImage(),
                        DataBuffer.TYPE_BYTE, "ssgci_bin");
                if (n_years > 1)
                    SoilSealingTestUtils.storeGeoTIFFSampleImage(beans, WIDTH, HEIGHT, beans.get(0).getCurrentImage(),
                            DataBuffer.TYPE_BYTE, "ssgci_bin2");
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "Could not save GeoTIFF Sample for testing", e);
            }
        }

        // I WOULD EXPECT A 2 X N_GEOMETRIS MATRIX
        // WHERE LINE 0 == INDEX VALUES FOR REFERENCE TIME
        // WHERE LINE 1 == INDEX VALUES FOR CURRENT TIME
        // double[][] result = new double[n_adm_units][n_years];
        // if (n_years == 2) {
        // some cuda code
        // }
        CUDAClass cuda = new CUDAClass();
        switch (soilSealingIndexType) {

        // ------- CORINE LAND COVER -------
        case COVERAGE_COEFFICIENT:
            LOGGER.warning("Not yet implemented");
            return null;

        case RATE_OF_CHANGE:
            LOGGER.warning("Not yet implemented");
            return null;

        case MARGINAL_LAND_TAKE:
            LOGGER.warning("Not yet implemented");
            return null;

        case URBAN_SPRAWL:
            LOGGER.warning("Not yet implemented");
            return null;

        // ------- IMPERVIOUSNESS -------
        case URBAN_DISPERSION: // urban dispersion [ccl+sum(hist(~polyMax))]

            // loop for each administrative unit
            for (int j = 0; j < n_adm_units; j++) { // lauch for administrative units
                // number of grids per admin_unit to give in output:
                for (int i = 0; i < n_years; i++) { // launch for ref/curr/diff
                    boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                    result[j][i] = new double[1];
                    if (isFeasible) {
                        result[j][i][0] = cuda.urban_sprawl(beans, areaPix, i, j, Distribution);
                    } else
                        result[j][i][0] = Double.NaN;
                }
                /*
                 * if (n_years>1) { result[j][n_years][0] = result[j][1][0]-result[j][0][0]; }
                 */
            }
            return result;

        case EDGE_DENSITY: // edge density [perimeter]
            // loop for each administrative unit
            for (int j = 0; j < n_adm_units; j++) { // lauch for administrative units
                // number of grids per admin_unit to give in output:
                for (int i = 0; i < n_years; i++) { // launch for ref/curr/diff
                    boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                    result[j][i] = new double[1];
                    if (isFeasible)
                        result[j][i][0] = CUDAClass.edge_density(beans, areaPix, i, j);
                    else
                        result[j][i][0] = Double.NaN;
                }
            }
            return result;

        case DISPERSIVE_URBAN_GROWTH: // urban diffusion
            // loop for each administrative unit
            for (int j = 0; j < n_adm_units; j++) { // lauch for administrative units
                // number of grids per admin_unit to give in output:
                for (int i = 0; i < n_years; i++) { // launch for ref/curr/diff
                    boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                    result[j][i] = new double[1];
                    if (isFeasible) {
                        if (subId == SoilSealingSubIndexType.URBAN_AREA) { // urban_area [reduction]* --> a modification from original occurred!
                            result[j][i][0] = cuda.urban_area(beans, areaPix, i, j);
                        } else if (subId == SoilSealingSubIndexType.HIGHEST_POLYGON_RATIO) { // area of polygon with maximum extension
                                                                                             // [ccl+ave(hist(~polyMax))]
                            result[j][i][0] = cuda.highest_polygon_ratio(beans, areaPix, i, j);
                        } else if (subId == SoilSealingSubIndexType.OTHER_POLYGONS_RATIO) { // average area of other polygons [ccl+hist]
                            result[j][i][0] = cuda.others_polygons_avesurf(beans, areaPix, i, j);
                        }
                    } else {
                        result[j][i] = new double[] { Double.NaN };
                    }
                }
            }
            return result;
        case FRAGMENTATION: // FRAGMENTATION [myFragProg]
            boolean NEW_SCENARIO=false;
            // double[][][] result = new double[n_adm_units][3][];
            /*
             * result = [admin(i)] [year=3] [maplen=HEIGHT(i)*WIDTH(i)]
             */
            // loop for each administrative unit
            for (int j = 0; j < n_adm_units; j++) { // lauch for administrative units
                // number of grids per admin_unit to give in output:
                for (int i = 0; i < n_years; i++) { // launch for ref/curr/diff
                    boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                    if (isFeasible)
                        result[j][i] = CUDAClass.fragmentation(beans, rural, ray_pixels, i, j, NEW_SCENARIO);
                    else
                        result[j][i] = new double[] { Double.NaN };
                }
                if (n_years > 1) {
                    // TODO: result[j][n_years] = CUDAClass.mapDifference(beans, result[j][0], result[j][1]);
                    if (result[j][0][0] == Double.NaN || result[j][1][0] == Double.NaN) {
                        result[j][n_years] = new double[] { Double.NaN };
                    } else {
                        result[j][n_years] = new double[result[j][0].length];
                        for (int ii = 0; ii < result[j][0].length; ii++) {
                            double refValue = (result[j][0][ii] != SoilSealingImperviousnessProcess.FRAG_NODATA ? result[j][0][ii] : 0d);
                            double curValue = (result[j][1][ii] != SoilSealingImperviousnessProcess.FRAG_NODATA ? result[j][1][ii] : 0d);

                            result[j][n_years][ii] = (curValue - refValue >= 0 ? curValue - refValue : SoilSealingImperviousnessProcess.FRAG_NODATA);
                        }
                    }
                }
            }
            return result;
        case LAND_TAKE: // LAND TAKE
            for (int j = 0; j < n_adm_units; j++) {
                final List<double[]> resultCuda = CUDAClass.land_take(beans, j);
                // rearrange to fit the middleware process and GUI code requirements
                /*
                 * result[j][0] = resultCuda.get(0);// MAP result[j][1] = resultCuda.get(1);// COUNTS "+1" [ha] result[j][2] = resultCuda.get(1);//
                 * COUNTS "-1" [ha]
                 */

                // number of grids per admin_unit to give in output:
                result[j][0] = resultCuda.get(0);// MAP
            }
            return result;
        case POTENTIAL_LOSS_FOOD_SUPPLY: // potential loss of food supply
            for (int j = 0; j < n_adm_units; j++) {
                // rearrange to fit the middleware process and GUI code requirements
                result[j][0] = new double[1];
                result[j][0][0] = Math.round(CUDAClass.potloss_foodsupply(beans, j));
            }
            return result;
        case MODEL_URBAN_DEVELOPMENT:
            // loop for each administrative unit
            for (int j = 0; j < n_adm_units; j++) { // lauch for administrative units
                // number of grids per admin_unit to give in output:
                for (int i = 0; i < n_years; i++) { // launch for ref/curr/diff
                    boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                    result[j][i] = new double[3];
                    if (isFeasible) {
                        result[j][i] = cuda.modelUrbanDevelopment(beans, areaPix, i, j); // [X,Y1,Y2]
                        for (int k = 0; k < result[j][i].length; k++) {
                            if (Double.isInfinite(result[j][i][k])
                                    || Double.isNaN(result[j][i][k])) {
                                result[j][i][k] = 0.0;
                            }
                        }
                    } else
                        result[j][i] = new double[] { 0.0, 0.0, 0.0 };
                }
            }
            return result;
        case NEW_URBANIZATION: // LABEL = "Simulate new urbanization", must be run after fragmentation?
        case NEW_ECO_CORRIDOR: // LABEL = "Simulate new ecological corridor", must be run after fragmentation?
            int i = 0; // fictitious year
            ray_pixels = 10;// I should parameterize this
            for (int j = 0; j < n_adm_units; j++) {
                // NOTE:
                // This index should be run after fragmentation using one year!!
                // --> this way I take rural & ray_pixels/RADIUS from previous run! (I can anyway set default values for both)
                // --> I should use also the same beans as before

                // I have to wrap any following class in a dedicated class for new urbanization process
                // in order to calculate the required specific inputs!!
                boolean isFeasible = CUDAClass.SUT(beans, i, j) > 0;
                if (isFeasible) {
                    result[j][i] = 
                    CUDAClass.newUrbanization(beans, rural, ray_pixels, i, j, soilSealingIndexType == SoilSealingIndexType.NEW_URBANIZATION);
                } else {
                    result[j][i] = new double[] { Double.NaN };
                }
            }
            return result;
        default:
            break;
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
     * @param soilSealingIndexType 
     * @param waterBodiesMaskReference 
     * @throws IOException
     * @throws MismatchedDimensionException
     * @throws TransformException
     */
    private void populateBean(CUDABean bean, boolean reference, GridCoverage2D coverage,
            Geometry geo, MathTransform transform, int buffer, SoilSealingIndexType soilSealingIndexType, 
            FeatureTypeInfo waterBodiesMaskReference)
            throws IOException, MismatchedDimensionException, TransformException {

        RenderedImage image = coverage.getRenderedImage();

        // 0) Check if a buffer must be applied
        Geometry originalGeo = (Geometry) geo.clone();
        if (buffer > 0) {
            try {
                if (!"EPSG:4326".equals(
                        CRS.lookupIdentifier(coverage.getCoordinateReferenceSystem(), false))) {
                    geo = geo.buffer(buffer);
                } else {
                    geo = geo.buffer(buffer / SoilSealingProcessingUtils.DEGREES_TO_METER_RATIO);
                }
            } catch (FactoryException e) {
                geo = geo.buffer(buffer);
            }
        }

        // 1) Crop the two coverages with the selected Geometry
        Envelope2D destinationEnvelope = null;
        try {
            CoordinateReferenceSystem geometryCrs = coverage.getCoordinateReferenceSystem();
            if (geo.getSRID() > 0) {
                geometryCrs = CRS.decode("EPSG:"+geo.getSRID());
                transform = CRS.findMathTransform(coverage.getCoordinateReferenceSystem(), geometryCrs, true);
                geo = JTS.transform(geo, transform);
                
                // Apply Mask if necessary
                Geometry mask = null;
                if (waterBodiesMaskReference != null) {
                    mask = SoilSealingProcessingUtils.getWBodiesMask(waterBodiesMaskReference);
                    
                    if (mask != null) {
                        final AffineTransform gridToWorldCorner = 
                                (AffineTransform) coverage.getGridGeometry().getGridToCRS2D(PixelOrientation.UPPER_LEFT);
                        
                        try {
                            mask = SoilSealingProcessingUtils.toReferenceCRS(mask, geometryCrs, gridToWorldCorner, false);
                        } catch (NoninvertibleTransformException e) {
                            LOGGER.log(Level.WARNING, e.getMessage(), e);
                        }
                        geo = geo.difference(mask);
                    }
                }
            }
            destinationEnvelope = JTS.getEnvelope2D(geo.getEnvelopeInternal(), geometryCrs);
        } catch (FactoryException e) {
            LOGGER.log(Level.SEVERE, "Selected Geometries cannot be reprojected to the source data SRS!", e);
            throw new RuntimeException("Selected Geometries cannot be reprojected to the source data SRS!", e);
        }
        
        // //
        // Check the intersection
        // //
        final GeneralEnvelope intersectionEnvelope = new GeneralEnvelope(coverage.getEnvelope());
        intersectionEnvelope.setCoordinateReferenceSystem(coverage.getCoordinateReferenceSystem());
        // intersect the envelopes
        intersectionEnvelope.intersect(destinationEnvelope);
        if (intersectionEnvelope.isEmpty()) {
            LOGGER.severe("Selected Geometries do not intersect the source data!");
            throw new RuntimeException("Selected Geometries do not intersect the source data!");
        }
        
        GridCoverage2D crop = CROP.execute(coverage, geo, null);
        transform = ProjectiveTransform.create(
                (AffineTransform) crop.getGridGeometry().getGridToCRS(PixelInCell.CELL_CORNER))
                .inverse();

        // 2) Extract the BufferedImage from each image
        image = crop.getRenderedImage();

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
            
            Geometry bufferedGeometry = geo;
            if(soilSealingIndexType == SoilSealingIndexType.NEW_URBANIZATION) {
                bufferedGeometry = geo.difference(originalGeo);
            } else if (soilSealingIndexType == SoilSealingIndexType.NEW_ECO_CORRIDOR) {
                bufferedGeometry = geo.difference(originalGeo);
            }
            Geometry rsFilter = JTS.transform(bufferedGeometry, transform);
            ROI roiGeo = new ROIGeometry(rs);
            ROI roiFilter = new ROIGeometry(rsFilter);

            // 5) Extract an array of data from the transformed ROI
            byte[] roiData = getROIData((buffer > 0 ? roiFilter : roiGeo), rectIMG);
            bean.setRoi(roiData);
            bean.setRoiObj(roiGeo);

            if (SoilSealingTestUtils.TESTING) {
                SoilSealingTestUtils.storeGeometryAsWKT(
                        bufferedGeometry, 
                        "roiGeometry_" + soilSealingIndexType.getName(), 
                        coverage.getCoordinateReferenceSystem()
                );
                
                SoilSealingTestUtils.storeGeoTIFFSampleImage(
                        JTS.bounds(bufferedGeometry, coverage.getCoordinateReferenceSystem()), 
                        coverage, 
                        image.getWidth(), 
                        image.getHeight(), 
                        roiData, 
                        DataBuffer.TYPE_BYTE, 
                        "roiData_" + soilSealingIndexType.getName());
            }
            
            
            // 6) Setting the Coverage data array
            bean.setReferenceCoverage(crop);
            bean.setReferenceImage(byteData);

            // 7) Setting the Image dimensions
            bean.setHeight(rectIMG.height);
            bean.setWidth(rectIMG.width);
            bean.setMinX(rectIMG.x);
            bean.setMinY(rectIMG.y);
        } else {
            // 6) Setting the Coverage data array
            bean.setReferenceCoverage(crop);
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
        if(data.length == 1 && Double.isNaN(data[0])) {
            LOGGER.severe("The index did not produce any ouput on the selected area!");
            throw new RuntimeException("The index did not produce any ouput on the selected area!");
        }
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

        private int cellSize;

        private double areaPx;

        private GridCoverage2D referenceCoverage;

        public byte[] getReferenceImage() {
            return referenceImage;
        }

        public void setReferenceCoverage(GridCoverage2D coverage) {
            this.referenceCoverage = coverage;
        }

        public GridCoverage2D getReferenceCoverage() {
            return this.referenceCoverage;
        }

        public int getCellSize() {
            return cellSize;
        }

        public double getAreaPx() {
            return areaPx;
        }

        public void setAreaPix(double areaPx) {
            this.areaPx = areaPx;
            this.cellSize = (int) Math.sqrt(areaPx);
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
