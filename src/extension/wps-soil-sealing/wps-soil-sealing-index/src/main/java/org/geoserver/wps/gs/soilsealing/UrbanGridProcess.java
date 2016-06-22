/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import it.geosolutions.jaiext.algebra.AlgebraCRIF;
import it.geosolutions.jaiext.algebra.AlgebraDescriptor;
import it.geosolutions.jaiext.algebra.AlgebraDescriptor.Operator;
import it.geosolutions.jaiext.bandmerge.BandMergeCRIF;
import it.geosolutions.jaiext.bandmerge.BandMergeDescriptor;
import it.geosolutions.jaiext.buffer.BufferDescriptor;
import it.geosolutions.jaiext.buffer.BufferRIF;

import java.awt.RenderingHints;
import java.awt.image.DataBuffer;
import java.awt.image.RenderedImage;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.media.jai.JAI;
import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;
import javax.media.jai.operator.BandSelectDescriptor;

import org.geoserver.catalog.FeatureTypeInfo;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingSubIndexType;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.data.DataStore;
import org.geotools.data.DefaultTransaction;
import org.geotools.data.FeatureReader;
import org.geotools.data.FeatureSource;
import org.geotools.data.Query;
import org.geotools.data.Transaction;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.factory.GeoTools;
import org.geotools.geometry.jts.JTS;
import org.geotools.image.jai.Registry;
import org.geotools.jdbc.JDBCDataStore;
import org.geotools.process.ProcessException;
import org.geotools.process.factory.DescribeParameter;
import org.geotools.process.factory.DescribeResult;
import org.geotools.process.gs.GSProcess;
import org.geotools.referencing.CRS;
import org.jaitools.imageutils.ROIGeometry;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.feature.type.FeatureType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory2;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

/**
 * This process calculates various indexes on the UrbanGrids. Indexes 5-6-7b-7c are calculated using Urban Grids as polygons. The other indexes are
 * calculated using Urban Grids as rasters. Operations on polygons are calculated by multiple threads.
 * 
 * The following hypotheses must be verified:
 * <ul>
 * <li>Input Geometries must be transformed to the Raster space for the indexes 7a-8-9-10, while must be on the UrbanGrids CRS for the other indexes;
 * </li>
 * <li>Coverages must be cropped to the active area.</li>
 * </ul>
 * 
 * 
 * @author geosolutions
 * 
 */
@SuppressWarnings("deprecation")
public class UrbanGridProcess implements GSProcess {
    private static final CoordinateReferenceSystem WGS84;

    /** Logger used for logging exceptions */
    public static final Logger LOGGER = Logger.getLogger(UrbanGridProcess.class.toString());

    /** Constant multiplier from square meters to ha */
    public static final double HACONVERTER = 0.0001;

    /** Header of the local Lambert-Equal Area projection */
    public static final String PROJ_HEADER = "PROJCS[\"Local area projection/ LOCAL_AREA_PROJECTION\",";

    /** Footer of the local Lambert-Equal Area projection */
    public static final String PROJ_FOOTER = ",PROJECTION[\"Lambert Azimuthal Equal Area\", AUTHORITY[\"EPSG\",\"9820\"]],"
            + "PARAMETER[\"latitude_of_center\", %LAT0%], PARAMETER[\"longitude_of_center\", %LON0%],"
            + "PARAMETER[\"false_easting\", 0.0], PARAMETER[\"false_northing\", 0.0],"
            + "UNIT[\"m\", 1.0], AXIS[\"Northing\", NORTH], AXIS[\"Easting\", EAST], AUTHORITY[\"EPSG\",\"3035\"]]";

    /** Base CRS of the local Lambert-Equal Area projection */
    public static final String GEOGCS_4326 = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,"
            + "AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],"
            + "UNIT[\"degree\",0.01745329251994328,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

    /** Final WKT of the local Lambert-Equal Area projection */
    public static final String PROJ_4326 = PROJ_HEADER + GEOGCS_4326 + PROJ_FOOTER;

    /** Default Pixel Area */
    private static final double PIXEL_AREA = 400;

    public static final String JAI_EXT_PRODUCT = "it.geosolutions.jaiext";
    static {
        try {
            Registry.registerRIF(JAI.getDefaultInstance(), new BufferDescriptor(), new BufferRIF(),
                    JAI_EXT_PRODUCT);
            Registry.registerRIF(JAI.getDefaultInstance(), new AlgebraDescriptor(),
                    new AlgebraCRIF(), JAI_EXT_PRODUCT);
            Registry.registerRIF(JAI.getDefaultInstance(), new BandMergeDescriptor(),
                    new BandMergeCRIF(), JAI_EXT_PRODUCT);
        } catch (Throwable e) {
            // swallow exception in case the op has already been registered.
        }
        CoordinateReferenceSystem crs = null;
        try {
            crs = CRS.decode("EPSG:4326");
        } catch (NoSuchAuthorityCodeException e) {
            LOGGER.log(Level.WARNING, e.getMessage(), e);
        } catch (FactoryException e) {
            LOGGER.log(Level.WARNING, e.getMessage(), e);
        }

        WGS84 = crs;
    }

    /** Countdown latch used for handling various threads simultaneously */
    private CountDownLatch latch;

    /** Imperviousness Vectorial Layer */
    private FeatureTypeInfo imperviousnessReference;

    /** Path associated to the shapefile of the reference image */
    private String referenceYear;

    /** Path associated to the shapefile of the current image */
    private String currentYear;

    @SuppressWarnings("unused")
    private String pathToRefShp;

    @SuppressWarnings("unused")
    private String pathToCurShp;

    public UrbanGridProcess(FeatureTypeInfo imperviousnessReference, String referenceYear,
            String currentYear) {
        this.imperviousnessReference = imperviousnessReference;
        this.referenceYear = referenceYear;
        this.currentYear = currentYear;
    }

    public UrbanGridProcess(String pathToRefShp, String pathToCurShp) {
        this.pathToRefShp = pathToRefShp;
        this.pathToCurShp = pathToCurShp;
    }

    // HP to verify
    // HP1 = admin geometries in Raster space, for index 7a-8-9-10; in SHP CRS for the other indexes
    // HP2 = Coverages already cropped and transformed to the Raster Space

    @DescribeResult(name = "UrbanGridProcess", description = "Urban Grid indexes", type = List.class)
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
            @DescribeParameter(name = "buffer", min = 0, description = "Buffer in meters") double buffer, 
            @DescribeParameter(name = "waterBodiesMaskReference", min = 0, description = "Name of the water bodies mask layer, optionally fully qualified (workspace:name)") FeatureTypeInfo waterBodiesMaskReference)
    
            throws IOException {

        // Checks on the index 7 "Dispersive Urban Growth"
        boolean nullSubId = subId == null;
        boolean subIndexA = !nullSubId && subId == SoilSealingSubIndexType.URBAN_AREA;
        boolean subIndexC = !nullSubId && subId == SoilSealingSubIndexType.HIGHEST_POLYGON_RATIO;
        boolean subIndexB = !nullSubId && subId == SoilSealingSubIndexType.OTHER_POLYGONS_RATIO;

        if (soilSealingIndexType == SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH
                && (nullSubId || !(subIndexA || subIndexB || subIndexC))) {
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

        // Number of Geometries
        int numThreads = rois.size();
        // Check if Geometry area or perimeter must be calculated
        boolean area = false;
        // Simple class set used for the raster calculations on indexes 7a-9-10
        Set<Integer> classes = new TreeSet<Integer>();
        classes.add(Integer.valueOf(1));
        List<List<Integer>> constantMatrix = null;
        // Selection of the operation to do for each index
        switch (soilSealingIndexType) {
        case URBAN_DISPERSION:
            area = true;
            break;
        case EDGE_DENSITY:
            area = false;
            break;
        case DISPERSIVE_URBAN_GROWTH:
            area = true;
            // If index is 7a raster calculation can be executed
            if (subIndexA) {
                return new CLCProcess().execute(referenceCoverage, nowCoverage, classes,
                        SoilSealingIndexType.COVERAGE_COEFFICIENT, areaPx, rois, null, null, true);
            }
            break;
        case FRAGMENTATION:
            // Raster elaboration
            return prepareImages(referenceCoverage, nowCoverage, rois, areaPx * HACONVERTER, SoilSealingImperviousnessProcess.FRAG_NODATA);
            
        // For the indexes 9-10 Zonal Stats are calculated
        case LAND_TAKE:
            // I have to compute IMPERVIOUSNESS difference between two TIMES: Icurrent - Ireference
            // I do NOT need to divide by POPULATION difference (Pcurrent - Preference)
            /*
             * return new CLCProcess().execute(referenceCoverage, nowCoverage, classes, CLCProcess.THIRD_INDEX, areaPx, rois, populations,
             * Double.valueOf(1), null);
             */

            // AF: Task-B36; This index should not be relative to the population.
            if (populations == null) {
                constantMatrix = GetConstFilledMatrix(2, 1, Integer.valueOf(1));
            } else {
                constantMatrix = GetConstFilledMatrix(populations.size(), populations.get(0).size(),
                        Integer.valueOf(1));
            }
            return new CLCProcess().execute(referenceCoverage, nowCoverage, classes,
                    SoilSealingIndexType.MARGINAL_LAND_TAKE, areaPx, rois, constantMatrix,
                    Double.valueOf(1), null);
        case POTENTIAL_LOSS_FOOD_SUPPLY:
            // I have to compute IMPERVIOUSNESS difference between two TIMES: Icurrent - Ireference
            // I do NOT need to divide by POPULATION difference (Pcurrent - Preference)
            if (coeff != null) {
                /*
                 * return new CLCProcess().execute(referenceCoverage, nowCoverage, classes, CLCProcess.THIRD_INDEX, areaPx, rois, populations, coeff,
                 * null);
                 */
                // AF: Task-B40; This index should not be relative to the population.
                if (populations == null) {
                    constantMatrix = GetConstFilledMatrix(2, 1, Integer.valueOf(1));
                } else {
                    constantMatrix = GetConstFilledMatrix(populations.size(),
                            populations.get(0).size(), Integer.valueOf(1));
                }
                return new CLCProcess().execute(referenceCoverage, nowCoverage, classes,
                        SoilSealingIndexType.MARGINAL_LAND_TAKE, areaPx, rois, constantMatrix,
                        coeff, null);
            } else {
                throw new IllegalArgumentException(
                        "No coefficient provided for the selected index");
            }
        case MODEL_URBAN_DEVELOPMENT:
        case NEW_URBANIZATION:
        case NEW_ECO_CORRIDOR:
            area = true;
            break;
        default:
            throw new IllegalArgumentException("Wrong index declared");
        }

        // If index is not 7a-8-9-10 then the input Urban Grids must be loaded
        // from the shp file.

        double[] statsRef = null;
        double[] statsNow = null;
        double[][][] statsComplex = null;

        try {
            // For each coverage are calculated the results
            if (referenceCoverage != null && referenceYear != null
                    && imperviousnessReference != null) {
                statsRef = prepareResults(referenceYear, imperviousnessReference,
                        soilSealingIndexType, rois, subIndexB, numThreads, area);
            }

            if (nowCoverage != null && currentYear != null && imperviousnessReference != null) {
                statsNow = prepareResults(currentYear, imperviousnessReference,
                        soilSealingIndexType, rois, subIndexB, numThreads, area);
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new ProcessException(e);
        }
        // Result accumulation
        List<StatisticContainer> results = accumulateResults(rois, statsRef, statsNow,
                statsComplex);

        return results;

    }

    private List<List<Integer>> GetConstFilledMatrix(int size, int size2, Integer value) {
        final List<List<Integer>> constMatrix = new ArrayList<List<Integer>>();
        final Integer[] fill = new Integer[size2];
        Arrays.fill(fill, value);

        for (int i = 0; i < size; i++) {
            constMatrix.add(Arrays.asList(fill));
        }

        return constMatrix;
    }

    /**
     * Takes the in input the result for each Coverage and return the result as a List of {@link StatisticContainer} objects.
     * 
     * @param rois input geometries used for calculations
     * @param reference reference coverage results array
     * @param now current coverage results array
     * @param statsComplex
     * @return
     */
    protected List<StatisticContainer> accumulateResults(List<Geometry> rois, double[] reference,
            double[] now, double[][][] statsComplex) {
        // Geometries number
        int numGeo = rois.size();
        // Final list initialization
        List<StatisticContainer> results = new ArrayList<StatisticContainer>(numGeo);
        // Check on the input coverages
        if (reference == null && now == null && statsComplex == null) {

            throw new ProcessException("No result has been calculated");

        } else if (reference != null && now == null) {
            // check on the dimensions
            if (numGeo != reference.length) {
                throw new ProcessException(
                        "Geometries and their results don't have the same dimensions");
            }
            // For each Geometry a container is created
            for (int i = 0; i < numGeo; i++) {
                Geometry geo = rois.get(i);

                StatisticContainer container = new StatisticContainer(geo,
                        new double[] { reference[i] }, null, null);
                results.add(container);
            }
        } else if (reference == null && now != null) {
            // check on the dimensions
            if (numGeo != now.length) {
                throw new ProcessException(
                        "Geometries and their results don't have the same dimensions");
            }
            // For each Geometry a container is created
            for (int i = 0; i < numGeo; i++) {
                Geometry geo = rois.get(i);

                StatisticContainer container = new StatisticContainer(geo, new double[] { now[i] },
                        null, null);
                results.add(container);
            }
        } else if (reference == null && now == null && statsComplex != null) {
            // check on the dimensions
            if (numGeo != statsComplex.length) {
                throw new ProcessException(
                        "Geometries and their results don't have the same dimensions");
            }
            // For each Geometry a container is created
            for (int i = 0; i < numGeo; i++) {
                Geometry geo = rois.get(i);

                StatisticContainer container = new StatisticContainer(geo, null, null,
                        statsComplex[i]);
                results.add(container);
            }

        } else {
            // check on the dimensions
            /*
             * if (numGeo != now.length || numGeo != reference.length) { throw new ProcessException(
             * "Geometries and their results don't have the same dimensions"); }
             */

            // For each Geometry a container is created
            for (int i = 0; i < numGeo; i++) {
                // Selection of the geometry
                Geometry geo = rois.get(i);
                // Selection of the index for the current and reference coverages
                double nowIdx = now[i];
                double refIdx = reference[i];
                // Percentual variation calculation
                double percentual = ((nowIdx - refIdx) / refIdx) * 100;
                StatisticContainer container = new StatisticContainer();
                container.setGeom(geo);
                container.setResultsRef(new double[] { refIdx });
                container.setResultsNow(new double[] { nowIdx });
                container.setResultsDiff(new double[] { percentual });

                results.add(container);
            }
        }

        return results;
    }

    /**
     * Calculates the index for each Geometry and takes a ShapeFile path for Urban Grid Geometries.
     * 
     * @param year Year of the shapefile to use
     * @param imperviousnessReference Layer of the shapefile to use
     * @param soilSealingIndexType index to calculate
     * @param rois Input Administrative Areas
     * @param subIndexB Boolean indicating if the subIndex to calculate is "b"
     * @param numThreads Total number of Threads to lauch
     * @param area Boolean indicating if Urban Grid Area must be calculated
     * @return
     * @throws MalformedURLException
     * @throws IOException
     * @throws InterruptedException
     * @throws FactoryException
     * @throws TransformException
     */
    private double[] prepareResults(String year, FeatureTypeInfo imperviousnessReference,
            SoilSealingIndexType soilSealingIndexType, List<Geometry> rois, boolean subIndexB,
            int numThreads, boolean area) throws MalformedURLException, IOException,
            InterruptedException, FactoryException, TransformException {
        // Calculation on the Urban Grids
        List<ListContainer> urbanGrids = calculateGeometries(year, imperviousnessReference, rois,
                numThreads, area);
        // Results
        double[] stats = new double[rois.size()];
        // Counter used for cycling on the Geometries
        int counter = 0;
        // Cycle on the urbanGrid results
        for (ListContainer container : urbanGrids) {

            if (area) {
                // List of all the areas
                List<Double> areas = container.getSortedList();
                if (areas != null && areas.size() > 0) {
                    // Total polygon number except the biggest
                    int numPolyNotMax = areas.size() - 1;
                    // Area of the maximum polygon
                    double polyMaxArea = areas.get(numPolyNotMax);
                    // Calculation of the total urban area
                    double sut = container.getTotalArea();
                    // Calculation of the urban area without the maximum polygon area
                    double sud = sut - polyMaxArea;

                    // Calculation of the indexes
                    switch (soilSealingIndexType) {
                    case URBAN_DISPERSION:
                        stats[counter] = (sud / sut) * 100.0;
                        break;
                    case DISPERSIVE_URBAN_GROWTH:
                        // Check on the subIndex selected
                        if (subIndexB) {
                            stats[counter] = (polyMaxArea / sut) * 100.0;
                        } else {
                            stats[counter] = (sud / numPolyNotMax) * HACONVERTER;
                        }
                    default:
                        break;
                    }
                } else {
                    stats[counter] = 0;
                }
            } else {
                // Selection of the Geometry
                Geometry geo = rois.get(counter);
                // Selection of the Geometry CRS
                CoordinateReferenceSystem sourceCRS = CRS.decode("EPSG:" + geo.getSRID());
                // Geometry reprojection
                Geometry geoPrj = reprojectToEqualArea(sourceCRS, geo);
                if (geoPrj == null) {
                    throw new ProcessException(
                            "Unable to reproject the input Administrative Geometry");
                }
                // Geometry Area
                double areaAdmin = geoPrj.getArea();
                // Index calculations
                stats[counter] = (container.getTotalPerimeter() / areaAdmin) / HACONVERTER;
            }
            // Counter update
            counter++;
        }

        return stats;
    }

    /**
     * Calculates the UrbanGrid area/perimeters for each Administrative area inside a separate thread.
     * 
     * @param year Input ShapeFile year
     * @param imperviousnessReference Input ShapeFile layer
     * @param rois List of all the input Geometries
     * @param numThreads Number of threads to lauch
     * @param area Boolean indicating if area must be calculated. (Otherwise perimeter is calculated)
     * @return
     * @throws MalformedURLException
     * @throws IOException
     * @throws InterruptedException
     */
    private List<ListContainer> calculateGeometries(String year,
            FeatureTypeInfo imperviousnessReference, List<Geometry> rois, int numThreads,
            boolean area) throws MalformedURLException, IOException, InterruptedException {
        // Initialization of the CountDown latch for handling multiple threads together
        latch = new CountDownLatch(numThreads);
        // Datastore creation
        final JDBCDataStore ds = (JDBCDataStore) imperviousnessReference.getStore()
                .getDataStore(null);
        // ThreadPoolExecutor object used for launching multiple threads simultaneously
        ThreadPoolExecutor executor = new ThreadPoolExecutor(numThreads, numThreads, 60,
                TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(1000000));
        // Final list containing the result calculated by each thread
        List<ListContainer> allLists = new ArrayList<ListContainer>(numThreads);
        // Cycle on the input geometries
        for (Geometry geo : rois) {
            // Creation of a new ListContainer object
            ListContainer container = new ListContainer();
            allLists.add(container);
            // Creation of a new Runnable for the UrbanGrids computation
            MyRunnable run = new MyRunnable(year, geo, imperviousnessReference, ds, container,
                    area);
            executor.execute(run);
        }
        // Waiting until all the threads have finished
        latch.await();
        // Executor closure
        executor.shutdown();

        executor.awaitTermination(30, TimeUnit.SECONDS);

        // Datastore disposal
        // ds.dispose();

        return allLists;
    }

    /**
     * Private method which reprojects the input Geometry in the input CRS to a Lambert-Equal Area CRS used for calculating Geometry Area and
     * perimeter.
     * 
     * @param sourceCRS Source geometry CRS.
     * @param sourceGeometry Source Geometry
     * @return
     * @throws FactoryException
     * @throws TransformException
     */
    private Geometry reprojectToEqualArea(CoordinateReferenceSystem sourceCRS,
            Geometry sourceGeometry) throws FactoryException, TransformException {
        // Reproject to the Lambert Equal Area
        // Geometry center used for centering the reprojection on the Geometry(reduces distance artifacts)
        Point center = sourceGeometry.getCentroid();
        // Creation of the MathTransform associated to the reprojection
        MathTransform transPoint = CRS.findMathTransform(sourceCRS, WGS84, true);
        Point centerRP = (Point) JTS.transform(center, transPoint);
        // Creation of a wkt for the selected Geometry
        String wkt = PROJ_4326.replace("%LAT0%", String.valueOf(centerRP.getY()));
        wkt = wkt.replace("%LON0%", String.valueOf(centerRP.getX()));
        // Parsing of the selected WKT
        final CoordinateReferenceSystem targetCRS = CRS.parseWKT(wkt);
        // Creation of the MathTransform associated to the reprojection
        MathTransform trans = CRS.findMathTransform(sourceCRS, targetCRS);
        // Geometry reprojection
        Geometry geoPrj;
        if (!trans.isIdentity()) {
            geoPrj = JTS.transform(sourceGeometry, trans);
        } else {
            geoPrj = sourceGeometry;
        }
        return geoPrj;
    }

    /**
     * Private method used for calculating index 8. This method takes 1/2 coverages in input and counts the not-0 values on a buffer for each pixel.
     * If 2 coverages are provided the result will return the image for each coverage and their difference.
     * 
     * @param referenceCoverage Input reference coverage
     * @param nowCoverage Input current coverage
     * @param geoms Input Administrative Areas
     * @param pixelArea Pixel area
     * @param destinationNoData 
     * @return
     */
    private List<StatisticContainer> prepareImages(GridCoverage2D referenceCoverage,
            GridCoverage2D nowCoverage, List<Geometry> geoms, double pixelArea, double destinationNoData) {
        // Boolean indicating the presence of the reference and current coverages
        boolean refExists = referenceCoverage != null;
        boolean nowExists = nowCoverage != null;
        // Selections of the Hints to use
        RenderingHints hints = GeoTools.getDefaultHints().clone();

        RenderedImage inputImage = null;
        // Merging of the 2 images if they are both present or selection of the single image
        if (refExists) {
            if (nowExists) {
                inputImage = BandMergeDescriptor.create(
                        null, 
                        destinationNoData, 
                        hints,
                        referenceCoverage.getRenderedImage(), 
                        nowCoverage.getRenderedImage());
            } else {
                inputImage = referenceCoverage.getRenderedImage();
            }
        } else {
            inputImage = nowCoverage.getRenderedImage();
        }
        // ROI preparation
        List<ROI> rois = new ArrayList<ROI>(geoms.size());
        // ROIGeometry associated to the geometry objects
        for (Geometry geom : geoms) {
            rois.add(new ROIGeometry(geom));
        }

        // Padding dimensions used for the buffer creation
        int leftPad = 10;
        int rightPad = 10;
        int bottomPad = 10;
        int topPad = 10;

        // Final list initialization
        List<StatisticContainer> stats = new ArrayList<StatisticContainer>(1);

        StatisticContainer container = new StatisticContainer();

        // Buffer calculation
        RenderedOp buffered = BufferDescriptor.create(inputImage, BufferDescriptor.DEFAULT_EXTENDER,
                leftPad, rightPad, topPad, bottomPad, rois, null, destinationNoData, null,
                DataBuffer.TYPE_DOUBLE, pixelArea, hints);
        // Selection of the first image
        RenderedOp imageRef = BandSelectDescriptor.create(buffered, new int[] { 0 }, hints);
        // Setting of the first image
        container.setReferenceImage(imageRef);

        RenderedOp imageNow = null;
        // if even the current coverage exists, it is taken.
        if (buffered.getNumBands() > 1) {
            imageNow = BandSelectDescriptor.create(buffered, new int[] { 1 }, hints);

            container.setNowImage(imageNow);
            // Calculation of the variation between current and reference images
            RenderedOp diff = AlgebraDescriptor.create(
                    Operator.SUBTRACT, 
                    null, 
                    null, 
                    destinationNoData,
                    hints, 
                    imageNow, 
                    imageRef);

            container.setDiffImage(diff);
        }
        // Storing of the result
        stats.add(container);

        return stats;
    }

    /**
     * This class implements Runnable and is used for executing calculations on the UrbanGrids for each Geometry.
     */
    class MyRunnable implements Runnable {

        private String year;

        private Geometry geo;

        private FeatureTypeInfo imperviousnessReference;

        private DataStore ds;

        private ListContainer values;

        private final boolean area;

        public MyRunnable(String year, Geometry geo, FeatureTypeInfo imperviousnessReference,
                DataStore ds, ListContainer values, boolean area) {
            this.year = year;
            this.geo = geo;
            this.imperviousnessReference = imperviousnessReference;
            this.ds = ds;
            this.values = values;
            this.area = area;
        }

        @SuppressWarnings("rawtypes")
        @Override
        public void run() {
            // Selection of the Feature Source associated to the datastore
            String typeName = null;
            FeatureSource source = null;
            try {
                typeName = imperviousnessReference.getFeatureType().getName().getLocalPart();
                source = ds.getFeatureSource(typeName);
            } catch (IOException e) {
                LOGGER.log(Level.SEVERE, e.getMessage());
                throw new ProcessException(e);
            }
            // If the Feature source is not present, an exception is thrown
            if (source == null) {
                throw new ProcessException("Source datastore not found");
            }

            FeatureType schema = source.getSchema();

            // usually "THE_GEOM" for shapefiles
            String geometryPropertyName = schema.getGeometryDescriptor().getLocalName();
            if (geometryPropertyName == null || geometryPropertyName.isEmpty()) {
                geometryPropertyName = "THE_GEOM";
            }

            // ShapeFile CRS
            CoordinateReferenceSystem sourceCRS = schema.getGeometryDescriptor()
                    .getCoordinateReferenceSystem();
            // Filter on the data store by selecting only the geometries contained into the input Geometry
            FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2(null);

            Filter yearFilter = ff.equals(ff.property("imp_year"), ff.literal(this.year));
            Filter geometryFilter = ff.within(ff.property(geometryPropertyName), ff.literal(geo));
            Filter queryFilter = ff.and(Arrays.asList(yearFilter, geometryFilter));
            Query query = new Query(typeName, queryFilter);
            // Feature collection selection
            FeatureReader<SimpleFeatureType, SimpleFeature> ftReader = null;
            Transaction transaction = new DefaultTransaction();
            try {
                ftReader = ds.getFeatureReader(query, transaction);
            } catch (IOException e) {
                LOGGER.log(Level.SEVERE, e.getMessage());
                throw new ProcessException(e);
            }

            if (ftReader == null) {
                throw new ProcessException("Source Feature collection not found");
            }

            double totalPerimeter = 0;

            // Selection of the inner list
            List<Double> areas = new ArrayList<Double>();

            double totalArea = 0;
            // Cycle on each polygon
            try {
                while (ftReader.hasNext()) {
                    SimpleFeature feature = (SimpleFeature) ftReader.next();
                    Geometry sourceGeometry = (Geometry) feature.getDefaultGeometry();
                    // If the geometry is a Polygon, then the operations are executed
                    // reprojection of the polygon
                    Geometry geoPrj = reprojectToEqualArea(sourceCRS, sourceGeometry);
                    // Area/Perimeter calculation
                    if (geoPrj != null) {
                        if (area) {
                            double area = geoPrj.getArea();
                            areas.add(area);
                            totalArea += area;
                        } else {
                            totalPerimeter += geoPrj.getLength();
                        }
                    }

                }

            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, e.getMessage());
                throw new ProcessException(e);
            } finally {
                try {
                    // Iterator closure
                    if (ftReader != null) {
                        ftReader.close();
                    }

                    transaction.commit();
                    transaction.close();
                } catch (Exception e) {
                    LOGGER.log(Level.WARNING, e.getMessage());
                }
            }
            // Saving results
            if (area) {
                values.setList(areas);
                values.setTotalArea(totalArea);
            } else {
                values.setTotalPerimeter(totalPerimeter);
            }
            // Countdown of the latch
            latch.countDown();
        }
    }

    /**
     * Container class used for passing parameters between threads.
     */
    class ListContainer {
        /** List containing all the polygon areas */
        private List<Double> list;

        /** Sum of all the areas */
        private Double totalArea;

        /** Sum of all the perimeters */
        private Double totalPerimeter;

        ListContainer() {
        }

        /** Return a sorted list of areas */
        public List<Double> getSortedList() {
            return list;
        }

        /**
         * Note: this method takes the list and sort it
         * 
         * @param list
         */
        public void setList(List<Double> list) {
            Collections.sort(list);
            this.list = list;
        }

        public double getTotalArea() {
            return totalArea;
        }

        public void setTotalArea(double totalArea) {
            this.totalArea = totalArea;
        }

        public Double getTotalPerimeter() {
            return totalPerimeter;
        }

        public void setTotalPerimeter(Double totalPerimeter) {
            this.totalPerimeter = totalPerimeter;
        }

        public void setTotalArea(Double totalArea) {
            this.totalArea = totalArea;
        }
    }
}
