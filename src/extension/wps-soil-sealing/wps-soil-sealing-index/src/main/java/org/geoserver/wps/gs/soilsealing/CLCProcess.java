/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import it.geosolutions.jaiext.bandmerge.BandMergeCRIF;
import it.geosolutions.jaiext.bandmerge.BandMergeDescriptor;
import it.geosolutions.jaiext.stats.Statistics;
import it.geosolutions.jaiext.stats.Statistics.StatsType;
import it.geosolutions.jaiext.zonal.ZonalStatsDescriptor;
import it.geosolutions.jaiext.zonal.ZonalStatsRIF;
import it.geosolutions.jaiext.zonal.ZoneGeometry;

import java.awt.image.DataBufferByte;
import java.awt.image.RenderedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.media.jai.JAI;
import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;

import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.factory.GeoTools;
import org.geotools.image.jai.Registry;
import org.geotools.process.factory.DescribeParameter;
import org.geotools.process.factory.DescribeResult;
import org.geotools.process.gs.GSProcess;
import org.jaitools.imageutils.ROIGeometry;

import com.vividsolutions.jts.geom.Geometry;

/**
 * This class calculates the indexes 1-2-3-4 from the input coverages. The user should pass the requested parameters and the index number and the
 * result is saved into a List of {@link StatisticContainer} objects, each of them stores the results for each Geometry.
 * 
 * The following hypotheses must be verified:
 * <ul>
 * <li>Geometries must be transformed to the Raster space;</li>
 * <li>Coverages must be cropped to the active area.</li>
 * </ul>
 * 
 * @author geosolutions
 * 
 */
@SuppressWarnings("deprecation")
public class CLCProcess implements GSProcess {
    /** Constant associated to the 0th idx */
    private static final int ZERO_IDX = 0;

    /** Default pixel area */
    public static final double PIXEL_AREA = 10000;

    /** Upper Bound used for index 4 */
    public static final double UPPER_BOUND_INDEX_4 = 1.5d;

    /** Lower Bound used for index 4 */
    public static final double LOWER_BOUND_INDEX_4 = 0.5d;

    public static final String JAI_EXT_PRODUCT = "it.geosolutions.jaiext";
    static {
        try {
            Registry.registerRIF(JAI.getDefaultInstance(), new BandMergeDescriptor(),
                    new BandMergeCRIF(), JAI_EXT_PRODUCT);
            Registry.registerRIF(JAI.getDefaultInstance(), new ZonalStatsDescriptor(),
                    new ZonalStatsRIF(), JAI_EXT_PRODUCT);
        } catch (Throwable e) {
            // swallow exception in case the op has already been registered.
        }
    }

    /**
     * 
     * Enum used for the 3-4 indexes
     * 
     */
    public enum IndexColor {
        GREEN(0), YELLOW(1), RED(2), BLUE(3);

        private final double value;

        IndexColor(double value) {
            this.value = value;
        }

        public double getValue() {
            return value;
        }

        public static IndexColor valueOf(double value) {
            return IndexColor.values()[(int) value];
        }

    }

    // HP to verify
    // HP1 = geometries in raster space
    // HP2 = Coverages already cropped

    @SuppressWarnings("unchecked")
    @DescribeResult(name = "CLCprocess", description = "CLC indexes", type = List.class)
    public List<StatisticContainer> execute(
            @DescribeParameter(name = "reference", description = "Name of the reference raster") GridCoverage2D referenceCoverage,
            @DescribeParameter(name = "now", description = "Name of the new raster") GridCoverage2D nowCoverage,
            @DescribeParameter(name = "classes", collectionType = Integer.class, min = 1, description = "The domain of the classes used in input rasters") Set<Integer> classes,
            @DescribeParameter(name = "index", min = 1, description = "Index to calculate") SoilSealingIndexType soilSealingIndexType,
            @DescribeParameter(name = "pixelarea", min = 0, description = "Pixel Area") Double pixelArea,
            @DescribeParameter(name = "rois", min = 1, description = "Administrative Areas") List<Geometry> rois,
            @DescribeParameter(name = "populations", min = 0, description = "Populations for each Area") List<List<Integer>> populations,
            @DescribeParameter(name = "coeff", min = 0, description = "Coefficient used in the 9-10 indexes calculations") Double coeff,
            @DescribeParameter(name = "percent", min = 0, description = "Indicator if the first index must be set in percentual") Boolean multiplier) {

        // First check on the number of input Coverages for the provided index
        boolean refExists = referenceCoverage != null;
        boolean nowExists = nowCoverage != null;

        if (soilSealingIndexType.getIdx() > 1 && (!nowExists || !refExists)) {
            throw new IllegalArgumentException("This index needs 2 input images");
        } else if (!nowExists && !refExists) {
            throw new IllegalArgumentException("No Coverages provided");
        }

        // Control on the population number for the 3° and 4° indexes
        int numAreas = rois.size();

        // PixelArea value
        double area = 0;
        if (pixelArea == null) {
            area = PIXEL_AREA;
        } else {
            area = pixelArea;
        }

        // Convert to Ha
        area *= UrbanGridProcess.HACONVERTER;

        // Check if percentual variation must be calculated
        boolean percentual = false;
        if (multiplier != null) {
            percentual = multiplier;
        }
        // Other check related to the indexes
        switch (soilSealingIndexType) {
            case COVERAGE_COEFFICIENT:
            case RATE_OF_CHANGE:
                break;
            case MARGINAL_LAND_TAKE:
            case URBAN_SPRAWL:
                int numPop = 0;
                if (populations != null) {
                    numPop = populations.size();
                }
                if (populations == null || numPop < 2) {
                    throw new IllegalArgumentException("Some Populations are not present");
                }
                int numPopRef = populations.get(ZERO_IDX).size();
                int numPopNow = populations.get(1).size();
                if (numAreas != numPopRef || numAreas != numPopNow) {
                    throw new IllegalArgumentException("Some Areas or Populations are not present");
                }
                break;
            default:
                throw new IllegalArgumentException("Wrong index selected");
        }

        RenderedImage inputImage = null;
        
        // Merging of the 2 images if they are both present or selection of the single image
        if (refExists) {
            if (nowExists) {
                double destinationNoData = 0d;
                inputImage = BandMergeDescriptor.create(null, destinationNoData,
                        GeoTools.getDefaultHints(), referenceCoverage.getRenderedImage(),
                        nowCoverage.getRenderedImage());
            } else {
                inputImage = referenceCoverage.getRenderedImage();
            }
        } else {
            inputImage = nowCoverage.getRenderedImage();
        }

        // Countercheck (Should never get here)
        if (inputImage == null) {
            throw new IllegalArgumentException("The image to calculate does not exists");
        }

        // Statistic object to calculate
        StatsType[] stats = new StatsType[] { StatsType.HISTOGRAM };

        // Further controls on the image band number and initialization of the statistics parameters
        int[] bands;
        double[] minBound;
        double[] maxBound;
        int[] numBins;
        // Band number of the input image
        int numBands = inputImage.getSampleModel().getNumBands();
        // Check if the band number is equal to 2
        boolean multiBanded = numBands == 2;
        if (multiBanded) {
            bands = new int[] { 0, 1 };
            minBound = new double[] { 0, 0 };
            maxBound = new double[] { 255, 255 };
            numBins = new int[] { 255, 255 };
        } else {
            bands = new int[] { 0 };
            minBound = new double[] { 0 };
            maxBound = new double[] { 255 };
            numBins = new int[] { 255 };
        }

        if (SoilSealingTestUtils.TESTING) {
            try {
                SoilSealingTestUtils.storeGeoTIFFSampleImage(
                        referenceCoverage.getEnvelope(),
                        referenceCoverage,
                        inputImage.getWidth(), 
                        inputImage.getHeight(), 
                        ((DataBufferByte)inputImage.getData().getDataBuffer()).getData(),
                        inputImage.getSampleModel().getDataType(), "ssgci__"+soilSealingIndexType.getName());
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        
        // Creation of a list of ROIs, each one for each Geometry object
        List<ROI> roilist = new ArrayList<ROI>();
        
        for (Geometry geom : rois) {
            roilist.add(new ROIGeometry(geom));
        }
        
        if (SoilSealingTestUtils.TESTING) {
            Geometry geometry = rois.get(0);
            
            // SoilSealingTestUtils.storeGeometryAsShapeFile(geometry, "ssgci__Coverage");
            SoilSealingTestUtils.storeGeometryAsWKT(geometry, "ssgci__"+soilSealingIndexType.getName(), referenceCoverage.getCoordinateReferenceSystem());
        }
        
        // Selection of the parameters
        RenderedOp zonalStats = ZonalStatsDescriptor.create(inputImage, null, null, roilist, null,
                null, false, bands, stats, minBound, maxBound, numBins, null, false, null);

        // Calculation of the results
        List<ZoneGeometry> results = (List<ZoneGeometry>) zonalStats.getProperty(ZonalStatsDescriptor.ZS_PROPERTY);

        // Class number
        int numClass = classes.size();

        // Zones counter
        int countZones = 0;

        // Result container
        List<StatisticContainer> container = new ArrayList<StatisticContainer>(numAreas);

        // Selection of the statistics for each Zone
        switch (soilSealingIndexType) {
            case COVERAGE_COEFFICIENT:
                // Elaboration for a 2-band image
                if (multiBanded) {
                    // For each Geometry
                    for (ZoneGeometry zone : results) {
                        // extraction of the statistics
                        double[][] coeffCop = calculateCoeffCop(classes, bands, zone, area, percentual);
                        // Geometry associated
                        Geometry geo = ((ROIGeometry) zone.getROI()).getAsGeometry();
                        // Variation array
                        double[] coeffVariation = calculateVariation(numClass, coeffCop, percentual);
                        // Object used for storing the index results
                        StatisticContainer statisticContainer = new StatisticContainer(geo,
                                coeffCop[ZERO_IDX], coeffCop[1], null);
                        statisticContainer.setResultsDiff(coeffVariation);
                        // Addition of the Statistics to a List
                        container.add(statisticContainer);
                    }
                } else {
                    // For each Geometry
                    for (ZoneGeometry zone : results) {
                        // extraction of the statistics
                        double[] coeffCop = calculateCoeffCop(classes, bands, zone, area,
                                percentual)[ZERO_IDX];
                        // Geometry associated
                        Geometry geo = ((ROIGeometry) zone.getROI()).getAsGeometry();
                        // Addition of the Statistics to a List
                        container.add(new StatisticContainer(geo, coeffCop, null, null));
                    }
                }
                break;
            case RATE_OF_CHANGE:
                // For each Geometry
                for (ZoneGeometry zone : results) {
                    double[][] coeffCop = new double[2][numClass];
                    // Cycle on the bands
                    for (int b = 0; b < numBands; b++) {
                        // extraction of the statistics
                        Statistics out = zone.getStatsPerBandNoClassifierNoRange(b)[ZERO_IDX];
    
                        double[] histogram = (double[]) out.getResult();
                        int count = 0;
                        // Storing of all the areas inside the array
                        for (Integer clc : classes) {
                            double clcArea = histogram[clc] * area;
                            coeffCop[b][count++] = clcArea;
                        }
                    }
                    // Calculation of the variation array
                    double[] coeffVariation = calculateVariation(numClass, coeffCop, true);
                    // Geometry associated
                    Geometry geo = ((ROIGeometry) zone.getROI()).getAsGeometry();
                    // Addition of the Statistics to a List
                    container.add(new StatisticContainer(geo, coeffVariation, null, null));
                }
                break;
            case MARGINAL_LAND_TAKE:
                // For each Geometry
                for (ZoneGeometry zone : results) {
                    // Calculation of the sum of all the areas
                    double[] consMarg = calculateCLCSum(classes, bands, zone, area);
                    // Calculation of the index
                    double first = consMarg[ZERO_IDX];
                    double second = consMarg[1];
                    double areaVar = (second - first);
    
                    double firstPop = populations.get(ZERO_IDX).get(countZones);
                    double secondPop = populations.get(1).get(countZones);
                    double popVar = (secondPop - firstPop);
                    // Geometry associated
                    Geometry geo = ((ROIGeometry) zone.getROI()).getAsGeometry();
    
                    double result = 0;
    
                    IndexColor color = null;
    
                    if (coeff != null) {
                        result = (areaVar / popVar) * coeff;
                    } else {
                        result = areaVar / popVar;
                    }
    
                    // Index result
                    if (areaVar >= 0) {
                        if (popVar >= 0) {
                            color = IndexColor.GREEN;
                        } else {
                            color = IndexColor.YELLOW;
                        }
                    } else {
                        if (popVar >= 0) {
                            color = IndexColor.RED;
                        } else {
                            color = IndexColor.BLUE;
                        }
                    }
    
                    // Addition of the Statistics to a List
                    StatisticContainer statisticContainer = new StatisticContainer(geo,
                            new double[] { result }, null, null);
    
                    // Setting of the color to use
                    if (color != null) {
                        statisticContainer.setColor(color);
                    }
                    container.add(statisticContainer);
                    // Update of the Zones
                    countZones++;
                }
                break;
            case URBAN_SPRAWL:
                for (ZoneGeometry zone : results) {
                    // Calculation of the sum of all the areas
                    double[] sumArray = calculateCLCSum(classes, bands, zone, area);
                    // Calculation of the index
                    double first = sumArray[ZERO_IDX];
                    double second = sumArray[1];
                    double areaTa = ((second - first) / first);
    
                    double firstPop = populations.get(ZERO_IDX).get(countZones);
                    double secondPop = populations.get(1).get(countZones);
                    double popTa = ((secondPop - firstPop) / firstPop);
                    // Geometry associated
                    Geometry geo = ((ROIGeometry) zone.getROI()).getAsGeometry();
    
                    double sprawl = areaTa / popTa;
    
                    IndexColor color = null;
    
                    if (sprawl > UPPER_BOUND_INDEX_4) {
                        color = IndexColor.RED;
                    } else if (sprawl < LOWER_BOUND_INDEX_4) {
                        color = IndexColor.GREEN;
                    } else {
                        color = IndexColor.YELLOW;
                    }
    
                    // Addition of the Statistics to a List
                    StatisticContainer statisticContainer = new StatisticContainer(geo,
                            new double[] { sprawl }, null, null);
    
                    // Setting of the color to use
                    if (color != null) {
                        statisticContainer.setColor(color);
                    }
                    container.add(statisticContainer);
    
                    // Update of the Zones
                    countZones++;
                }
                break;
            default:
                break;
        }
        
        return container;
    }

    /**
     * Computes the sum of all class areas, for each Geometry.
     * 
     * @param classes Set of all the classes to take into account
     * @param bands bands on which the elaborations must be executed
     * @param zone Object which contains the statistics for the Geometry
     * @param area pixel area
     */
    private double[] calculateCLCSum(Set<Integer> classes, int[] bands, ZoneGeometry zone,
            double area) {

        double[] consMarg = new double[bands.length];
        int numBands = bands.length;
        // Cycle on the bands
        for (int b = 0; b < numBands; b++) {
            // For each bands extracts the statistics
            Statistics out = zone.getStatsPerBandNoClassifierNoRange(b)[ZERO_IDX];
            // Calculation of the sum of all the areas
            double[] histogram = (double[]) out.getResult();
            for (Integer clc : classes) {
                double clcArea = histogram[clc] * area;
                consMarg[b] += clcArea;
            }
        }

        return consMarg;
    }

    /**
     * Private method used for the calculation of the variation of the 2 input indexes.
     * 
     * @param numClass number of the classes calculated
     * @param coeffCop array of the indexes each one for each time.
     * @param percentual boolean indicating if the variation must be calculated with percentual.
     */
    private double[] calculateVariation(int numClass, double[][] coeffCop, boolean percentual) {
        // Index multiplier value
        double multiplier = 1;

        if (percentual) {
            multiplier = 100;
        }

        double[] coeffVariation = new double[numClass];
        // Cycle on all the classes and calculation of the variation
        for (int i = 0; i < numClass; i++) {
            double first = coeffCop[0][i];
            double second = coeffCop[1][i];
            coeffVariation[i] = ((second - first) / first) * multiplier;
        }

        return coeffVariation;
    }

    /**
     * Private method used for extracting the results from each Geometry
     * 
     * @param classes input classes to calculate
     * @param bands bands to calculate
     * @param zone Zone object which stores the results
     * @param area pixel area
     * @param percentual boolean indicating if the result must be returned in percentual
     * @return
     */
    private double[][] calculateCoeffCop(Set<Integer> classes, int[] bands, ZoneGeometry zone,
            double area, boolean percentual) {
        // Result container
        double[][] coeffCop = new double[bands.length][classes.size()];
        // Index multiplier
        double multiplier = 1;
        // Setting the multiplier to 100 if requested
        if (percentual) {
            multiplier = 100;
        }
        // Cycle on all the bands
        double adminArea;
        int numBands = bands.length;
        for (int b = 0; b < numBands; b++) {
            // Selection of the statistics for the defined band
            Statistics out = zone.getStatsPerBandNoClassifierNoRange(b)[ZERO_IDX];
            // Calculation of the Administrative Unit Area by taking the pixel number and multiplying it per the pixel area
            adminArea = out.getNumSamples() * area;
            // Histogram of the classes
            double[] histogram = (double[]) out.getResult();
            // Cycle on all the classes in order to calculate the index for each of them
            int count = 0;
            for (Integer clc : classes) {
                double clcArea = histogram[clc] * area;
                coeffCop[b][count++] = clcArea / adminArea * multiplier;
            }
        }

        return coeffCop;
    }

    /**
     * Helper class used for storing the index results for each Geometry and passing it in output
     */
    public static class StatisticContainer {

        private Geometry geom;

        private double[] resultsRef;

        private double[] resultsNow;

        private double[] resultsDiff;

        private double[][] resultsComplex;

        private RenderedImage referenceImage;

        private RenderedImage nowImage;

        private RenderedImage diffImage;

        private IndexColor color;

        public StatisticContainer() {
        }

        public StatisticContainer(Geometry geom, double[] resultsRef, double[] resultsNow,
                double[][] resultsComplex) {
            this.geom = geom;
            this.resultsRef = resultsRef;
            this.resultsNow = resultsNow;
            this.resultsComplex = resultsComplex;
        }

        public Geometry getGeom() {
            return geom;
        }

        public void setGeom(Geometry geom) {
            this.geom = geom;
        }

        public double[] getResults() {
            return resultsRef;
        }

        public double[] getResultsRef() {
            return resultsRef;
        }

        public void setResultsRef(double[] resultsRef) {
            this.resultsRef = resultsRef;
        }

        public double[][] getResultsComplex() {
            return this.resultsComplex;
        }

        public void setResultsComplex(double[][] resultsComplex) {
            this.resultsComplex = resultsComplex;
        }

        public double[] getResultsNow() {
            return resultsNow;
        }

        public void setResultsNow(double[] resultsNow) {
            this.resultsNow = resultsNow;
        }

        public double[] getResultsDiff() {
            return resultsDiff;
        }

        public void setResultsDiff(double[] resultsDiff) {
            this.resultsDiff = resultsDiff;
        }

        public RenderedImage getReferenceImage() {
            return referenceImage;
        }

        public void setReferenceImage(RenderedImage referenceImage) {
            this.referenceImage = referenceImage;
        }

        public RenderedImage getNowImage() {
            return nowImage;
        }

        public void setNowImage(RenderedImage nowImage) {
            this.nowImage = nowImage;
        }

        public RenderedImage getDiffImage() {
            return diffImage;
        }

        public void setDiffImage(RenderedImage diffImage) {
            this.diffImage = diffImage;
        }

        public IndexColor getColor() {
            return color;
        }

        public void setColor(IndexColor color) {
            this.color = color;
        }
    }
}
