/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import it.geosolutions.jaiext.algebra.AlgebraDescriptor;
import it.geosolutions.jaiext.algebra.AlgebraDescriptor.Operator;
import it.geosolutions.jaiext.buffer.BufferDescriptor;
import it.geosolutions.jaiext.stats.Statistics;
import it.geosolutions.jaiext.stats.Statistics.StatsType;
import it.geosolutions.jaiext.stats.StatisticsDescriptor;

import java.awt.RenderingHints;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;
import javax.media.jai.TiledImage;

import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingSubIndexType;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridCoverageFactory;
import org.geotools.data.DataStore;
import org.geotools.data.DataStoreFinder;
import org.geotools.data.FeatureSource;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.factory.GeoTools;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.geotools.geometry.Envelope2D;
import org.geotools.geometry.jts.JTS;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.process.ProcessException;
import org.geotools.referencing.CRS;
import org.geotools.test.TestData;
import org.jaitools.imageutils.ROIGeometry;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.type.FeatureType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory2;
import org.opengis.geometry.Envelope;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LinearRing;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

public class UrbanGridProcessTest {

    public static final int DEF_H = 256;

    public static final int DEF_W = 256;

    public static final int DEF_TILE_H = 32;

    public static final int DEF_TILE_W = 32;

    public static final double DELTA = 0.1;

    private static CoordinateReferenceSystem crs;

    private static GridCoverage2D referenceCoverage;

    private static GridCoverage2D nowCoverage;

    private static ArrayList<Geometry> geomListRaster;

    private static ArrayList<Geometry> geomListUtm32N;

    private static UrbanGridProcess urbanProcess;

    private static String pathToRefShp = "src/test/resources/org/geoserver/wps/gs/soilsealing/test-data/referenceCov.shp";

    private static String pathToCurShp = "src/test/resources/org/geoserver/wps/gs/soilsealing/test-data/nowCov.shp";

    private static String refShp = "referenceCov.shp";

    private static String curShp = "nowCov.shp";

    private static double totalPerimeterRef;

    private static List<Double> areasRef;

    private static double totalAreaRef;

    private static double totalPerimeterCur;

    private static List<Double> areasCur;

    private static double totalAreaCur;

    private static double converter;

    private static double rasterGeoArea;

    private static double multiplier;

    private static List<List<Integer>> populations;

    public static final int DEFAULT_POP_REF = 10;

    public static final int DEFAULT_POP_NOW = 20;

    @BeforeClass
    public static void initialSetup()
            throws NoSuchAuthorityCodeException, FactoryException, IOException {
        crs = CRS.decode("EPSG:32632");

        referenceCoverage = createImage(true);
        nowCoverage = createImage(false);

        // Geometries in Raster space for indexes 7a-8-9-10
        geomListRaster = new ArrayList<Geometry>();

        GeometryFactory fact = new GeometryFactory();
        Coordinate[] coordinates = new Coordinate[5];
        for (int i = 0; i < coordinates.length; i++) {
            if (i == 0 || i == 4) {
                coordinates[i] = new Coordinate(0, 0);
            } else if (i == 3) {
                coordinates[i] = new Coordinate(DEF_W / 2, 0);
            } else if (i == 1) {
                coordinates[i] = new Coordinate(0, DEF_H / 2);
            } else {
                coordinates[i] = new Coordinate(DEF_W / 2, DEF_H / 2);
            }
        }
        LinearRing linear = new GeometryFactory().createLinearRing(coordinates);
        Polygon poly = new Polygon(linear, null, fact);

        geomListRaster.add(poly);

        // Geometries in UTM 32 N
        geomListUtm32N = new ArrayList<Geometry>();

        Coordinate[] coordinates2 = new Coordinate[5];
        for (int i = 0; i < coordinates2.length; i++) {
            if (i == 0 || i == 4) {
                coordinates2[i] = new Coordinate(0, DEF_H);
            } else if (i == 3) {
                coordinates2[i] = new Coordinate(0, DEF_H / 2);
            } else if (i == 1) {
                coordinates2[i] = new Coordinate(DEF_W / 2, DEF_H);
            } else {
                coordinates2[i] = new Coordinate(DEF_W / 2, DEF_H / 2);
            }
        }
        LinearRing linear2 = new GeometryFactory().createLinearRing(coordinates2);
        Polygon poly2 = new Polygon(linear2, null, fact);

        poly2.setSRID(32632);
        geomListUtm32N.add(poly2);

        urbanProcess = new UrbanGridProcess(pathToRefShp, pathToCurShp);

        // HAConverter
        converter = UrbanGridProcess.HACONVERTER;

        // Raster Geometry Area
        // rasterGeoArea = (DEF_W / 2 - 1)*(DEF_H / 2 - 1)*converter;
        rasterGeoArea = poly.getArea() * converter;

        // Reference
        totalAreaRef = 125 * converter;
        areasRef = new ArrayList<Double>(2);
        areasRef.add(25d * converter);
        areasRef.add(100d * converter);
        calculatePerimeters(poly2, true);
        // Current
        totalAreaCur = 115 * converter;
        areasCur = new ArrayList<Double>(2);
        areasCur.add(25d * converter);
        areasCur.add(90d * converter);
        calculatePerimeters(poly2, false);

        // Index 10 fake multiplier
        multiplier = 20d;

        // Pops
        populations = new ArrayList<List<Integer>>(2);
        List<Integer> popref = new ArrayList<Integer>(1);
        popref.add(DEFAULT_POP_REF);
        populations.add(popref);

        List<Integer> popnow = new ArrayList<Integer>(1);
        popnow.add(DEFAULT_POP_NOW);
        populations.add(popnow);
    }

    // @Test
    public void testIndex5() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.URBAN_DISPERSION, null, null, geomListUtm32N, null, null,
                false, 100, 100, null);

        // Expected results
        double sut = totalAreaRef;
        double sud = totalAreaRef - areasRef.get(areasRef.size() - 1);
        double expected = sud / sut;

        double calculated = results.get(0).getResults()[0];

        Assert.assertEquals(expected, calculated, DELTA);
    }

    // @Test
    public void testIndex5img2() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.URBAN_DISPERSION, null, null, geomListUtm32N, null, null,
                false, 100, 100, null);

        // Expected results Reference
        double sutRef = totalAreaRef;
        double sudRef = totalAreaRef - areasRef.get(areasRef.size() - 1);
        double expectedRef = sudRef / sutRef;

        double calculatedRef = results.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRef, calculatedRef, DELTA);

        // Expected results Current
        double sutCur = totalAreaCur;
        double sudCur = totalAreaCur - areasCur.get(areasCur.size() - 1);
        double expectedCur = sudCur / sutCur;

        double calculatedCur = results.get(0).getResultsNow()[0];

        Assert.assertEquals(expectedCur, calculatedCur, DELTA);
    }

    // @Test
    public void testIndex6() throws Exception {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.EDGE_DENSITY, null, null, geomListUtm32N, null, null, false,
                100, 100, null);
        // Expected Result Reference
        Geometry geoRpj = reprojectToEqualArea(crs, geomListUtm32N.get(0));
        double adminArea = geoRpj.getArea() * converter;
        double expectedRef = totalPerimeterRef / adminArea;

        double calculatedRef = results.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRef, calculatedRef, DELTA);
    }

    // @Test
    public void testIndex6img2() throws Exception {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.EDGE_DENSITY, null, null, geomListUtm32N, null, null, false,
                100, 100, null);
        // Expected Result Reference
        Geometry geoRpj = reprojectToEqualArea(crs, geomListUtm32N.get(0));
        double adminArea = geoRpj.getArea() * converter;
        double expectedRef = totalPerimeterRef / adminArea;

        double calculatedRef = results.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRef, calculatedRef, DELTA);

        // Expected Result Current
        double expectedCur = totalPerimeterCur / adminArea;

        double calculatedCur = results.get(0).getResultsNow()[0];

        Assert.assertEquals(expectedCur, calculatedCur, DELTA);
    }

    // @Test
    public void testIndex7() throws IOException {
        // Index 7a
        List<StatisticContainer> resultsA = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH, SoilSealingSubIndexType.URBAN_AREA,
                1d, geomListRaster, null, null, false, 100, 100, null);

        double expectedRefA = totalAreaRef / rasterGeoArea * 100;

        double calculatedRefA = resultsA.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefA, calculatedRefA, DELTA);

        // Index 7b

        List<StatisticContainer> resultsB = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH,
                SoilSealingSubIndexType.HIGHEST_POLYGON_RATIO, 1d, geomListUtm32N, null, null,
                false, 100, 100, null);

        double polyMaxArea = areasRef.get(areasRef.size() - 1);

        double expectedRefB = polyMaxArea / totalAreaRef * 100;

        double calculatedRefB = resultsB.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefB, calculatedRefB, DELTA);
        // Index 7c
        List<StatisticContainer> resultsC = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH,
                SoilSealingSubIndexType.OTHER_POLYGONS_RATIO, 1d, geomListUtm32N, null, null, false,
                100, 100, null);

        double polyAreaNotMax = totalAreaRef - polyMaxArea;

        double expectedRefC = polyAreaNotMax / (areasRef.size() - 1);

        double calculatedRefC = resultsC.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefC, calculatedRefC, DELTA);
    }

    // @Test
    public void testIndex7img2() throws IOException {
        // Index 7a
        List<StatisticContainer> resultsA = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH, SoilSealingSubIndexType.URBAN_AREA,
                1d, geomListRaster, null, null, false, 100, 100, null);
        // reference
        double expectedRefA = totalAreaRef / rasterGeoArea * 100;

        double calculatedRefA = resultsA.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefA, calculatedRefA, DELTA);

        // current
        double expectedCurA = totalAreaCur / rasterGeoArea * 100;

        double calculatedCurA = resultsA.get(0).getResultsNow()[0];

        Assert.assertEquals(expectedCurA, calculatedCurA, DELTA);

        // Index 7b

        List<StatisticContainer> resultsB = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH,
                SoilSealingSubIndexType.HIGHEST_POLYGON_RATIO, 1d, geomListUtm32N, null, null,
                false, 100, 100, null);
        // reference
        double polyMaxAreaRef = areasRef.get(areasRef.size() - 1);

        double expectedRefB = polyMaxAreaRef / totalAreaRef * 100;

        double calculatedRefB = resultsB.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefB, calculatedRefB, DELTA);

        // current
        double polyMaxAreaCur = areasCur.get(areasCur.size() - 1);

        double expectedCurB = polyMaxAreaCur / totalAreaCur * 100;

        double calculatedCurB = resultsB.get(0).getResultsNow()[0];

        Assert.assertEquals(expectedCurB, calculatedCurB, DELTA);

        // Index 7c
        List<StatisticContainer> resultsC = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH,
                SoilSealingSubIndexType.OTHER_POLYGONS_RATIO, 1d, geomListUtm32N, null, null, false,
                100, 100, null);

        // reference
        double polyAreaNotMaxRef = totalAreaRef - polyMaxAreaRef;

        double expectedRefC = polyAreaNotMaxRef / (areasRef.size() - 1);

        double calculatedRefC = resultsC.get(0).getResultsRef()[0];

        Assert.assertEquals(expectedRefC, calculatedRefC, DELTA);

        // current
        double polyAreaNotMaxCur = totalAreaCur - polyMaxAreaCur;

        double expectedCurC = polyAreaNotMaxCur / (areasCur.size() - 1);

        double calculatedCurC = resultsC.get(0).getResultsNow()[0];

        Assert.assertEquals(expectedCurC, calculatedCurC, DELTA);
    }

    @Test
    public void testIndex8() throws IOException {
        // Selections of the Hints to use
        RenderingHints hints = GeoTools.getDefaultHints().clone();
        int padding = 10;
        double destNoData = 0;
        double pixelArea = 1d;

        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.FRAGMENTATION, null, pixelArea, geomListRaster, null, null,
                false, 100, 100, null);

        List<ROI> rois = new ArrayList<ROI>(geomListRaster.size());

        for (Geometry geom : geomListRaster) {
            rois.add(new ROIGeometry(geom));
        }

        RenderedOp referenceExpected = BufferDescriptor.create(referenceCoverage.getRenderedImage(),
                BufferDescriptor.DEFAULT_EXTENDER, padding, padding, padding, padding, rois, null,
                destNoData, null, DataBuffer.TYPE_DOUBLE, pixelArea * converter, hints);

        RenderedImage referenceCalculated = results.get(0).getReferenceImage();
        // Check if the image is present
        Assert.assertNotNull(referenceCalculated);
        // Calculation of the variation between current and reference images
        RenderedOp diff = AlgebraDescriptor.create(Operator.SUBTRACT, null, null, destNoData, hints,
                referenceExpected, referenceCalculated);
        int[] bands = new int[] { 0 };
        int period = 1;
        StatsType[] stats = new StatsType[] { StatsType.MEAN };
        RenderedOp statsIMG = StatisticsDescriptor.create(diff, period, period, null, null, false,
                bands, stats, hints);

        Statistics[][] values = (Statistics[][]) statsIMG.getProperty(Statistics.STATS_PROPERTY);

        double meanCalculated = (Double) values[0][0].getResult();
        double meanExpected = 0;

        Assert.assertEquals(meanExpected, meanCalculated, DELTA);
    }

    @Test
    public void testIndex8img2() throws IOException {
        // Selections of the Hints to use
        RenderingHints hints = GeoTools.getDefaultHints().clone();
        int padding = 10;
        double destNoData = 0;
        double pixelArea = 1d;

        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.FRAGMENTATION, null, pixelArea, geomListRaster, null, null,
                false, 100, 100, null);

        RenderedImage referenceCalculated = results.get(0).getReferenceImage();
        RenderedImage currentCalculated = results.get(0).getNowImage();
        RenderedImage diffCalculated = results.get(0).getDiffImage();
        // Check if the image is present
        Assert.assertNotNull(referenceCalculated);
        Assert.assertNotNull(currentCalculated);
        Assert.assertNotNull(diffCalculated);

        List<ROI> rois = new ArrayList<ROI>(geomListRaster.size());

        for (Geometry geom : geomListRaster) {
            rois.add(new ROIGeometry(geom));
        }

        int[] bands = new int[] { 0 };
        int period = 1;
        StatsType[] stats = new StatsType[] { StatsType.MEAN };

        // Reference
        RenderedOp referenceExpected = BufferDescriptor.create(referenceCoverage.getRenderedImage(),
                BufferDescriptor.DEFAULT_EXTENDER, padding, padding, padding, padding, rois, null,
                destNoData, null, DataBuffer.TYPE_DOUBLE, pixelArea * converter, hints);

        // Calculation of the variation between current and reference images
        RenderedOp diff = AlgebraDescriptor.create(Operator.SUBTRACT, null, null, destNoData, hints,
                referenceExpected, referenceCalculated);

        RenderedOp statsIMG = StatisticsDescriptor.create(diff, period, period, null, null, false,
                bands, stats, hints);

        Statistics[][] values = (Statistics[][]) statsIMG.getProperty(Statistics.STATS_PROPERTY);

        double meanCalculated = (Double) values[0][0].getResult();
        double meanExpected = 0;

        Assert.assertEquals(meanExpected, meanCalculated, DELTA);

        // Current

        RenderedOp currentExpected = BufferDescriptor.create(nowCoverage.getRenderedImage(),
                BufferDescriptor.DEFAULT_EXTENDER, padding, padding, padding, padding, rois, null,
                destNoData, null, DataBuffer.TYPE_DOUBLE, pixelArea * converter, hints);

        // Calculation of the variation between current and reference images
        RenderedOp diffCur = AlgebraDescriptor.create(Operator.SUBTRACT, null, null, destNoData,
                hints, currentExpected, currentCalculated);

        RenderedOp statsIMGCur = StatisticsDescriptor.create(diffCur, period, period, null, null,
                false, bands, stats, hints);

        Statistics[][] valuesCur = (Statistics[][]) statsIMGCur
                .getProperty(Statistics.STATS_PROPERTY);

        double meanCalculatedCur = (Double) valuesCur[0][0].getResult();
        double meanExpectedCur = 0;

        Assert.assertEquals(meanExpectedCur, meanCalculatedCur, DELTA);
    }

    @Test
    public void testIndex9() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.LAND_TAKE, null, 1d, geomListRaster, populations, null, false,
                100, 100, null);

        double areaDiff = totalAreaCur - totalAreaRef;
        double deltaPop = DEFAULT_POP_NOW - DEFAULT_POP_REF;

        double expected = areaDiff / deltaPop;

        double calculated = results.get(0).getResults()[0];

        Assert.assertEquals(expected, calculated, DELTA);
    }

    @Test
    public void testIndex10() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, nowCoverage,
                SoilSealingIndexType.POTENTIAL_LOSS_FOOD_SUPPLY, null, 1d, geomListRaster,
                populations, multiplier, false, 100, 100, null);

        double areaDiff = totalAreaCur - totalAreaRef;
        double deltaPop = DEFAULT_POP_NOW - DEFAULT_POP_REF;

        double expected = areaDiff / deltaPop * multiplier;

        double calculated = results.get(0).getResults()[0];

        Assert.assertEquals(expected, calculated, DELTA);
    }

    // EXCEPTION TESTS
    @Test(expected = IllegalArgumentException.class)
    public void testNoCoverages() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(null, null,
                SoilSealingIndexType.POTENTIAL_LOSS_FOOD_SUPPLY, null, 1d, geomListRaster,
                populations, multiplier, false, 100, 100, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongSubId() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.DISPERSIVE_URBAN_GROWTH,
                SoilSealingSubIndexType.translate("d"), 1d, geomListRaster, populations, multiplier,
                false, 100, 100, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongIndex() throws IOException {
        List<StatisticContainer> results = urbanProcess.execute(referenceCoverage, null,
                SoilSealingIndexType.MODEL_URBAN_DEVELOPMENT,
                SoilSealingSubIndexType.translate("d"), 1d, geomListRaster, populations, multiplier,
                false, 100, 100, null);
    }

    @Test(expected = ProcessException.class)
    public void testNoDatastore() throws IOException {
        List<StatisticContainer> results = new UrbanGridProcess(refShp, curShp).execute(
                referenceCoverage, null, SoilSealingIndexType.URBAN_DISPERSION, null, 1d,
                geomListUtm32N, null, null, false, 100, 100, null);
    }

    @AfterClass
    public static void finalDispose() {
        referenceCoverage.dispose(true);
        nowCoverage.dispose(true);
    }

    public static GridCoverage2D createImage(boolean reference) {

        SampleModel sm = new ComponentSampleModel(DataBuffer.TYPE_BYTE, DEF_W, DEF_H, 1, DEF_W,
                new int[] { 0 });

        TiledImage img = new TiledImage(sm, DEF_TILE_W, DEF_TILE_H);

        int minX = 0;
        int maxX = 0;
        int minY = 0;
        int maxY = 0;

        if (reference) {
            minX = 10;
            maxX = 20;
            minY = 10;
            maxY = 20;
        } else {
            minX = 11;
            maxX = 20;
            minY = 10;
            maxY = 20;
        }

        int minPolStart = 0;
        int maxPolStart = 5;

        for (int i = minPolStart; i < maxPolStart; i++) {
            for (int j = minPolStart; j < maxPolStart; j++) {
                img.setSample(i, j, 0, 1);
            }
        }

        for (int i = minX; i < maxX; i++) {
            for (int j = minY; j < maxY; j++) {
                img.setSample(i, j, 0, 1);
            }
        }

        Envelope envelope = new Envelope2D(crs, 0, 0, DEF_W, DEF_H);

        GridCoverage2D result = new GridCoverageFactory(GeoTools.getDefaultHints()).create("test",
                img, envelope);

        return result;
    }

    private static void calculatePerimeters(Geometry geom, boolean reference) throws IOException {

        String path = null;

        if (reference) {
            path = refShp;
        } else {
            path = curShp;
        }

        // ShapeFile selection
        File file = TestData.file(UrbanGridProcessTest.class, path);

        Map map = new HashMap();
        map.put("url", file.toURL());
        // Datastore creation
        DataStore ds = DataStoreFinder.getDataStore(map);

        // Selection of the Feature Source associated to the datastore
        String typeName = null;
        FeatureSource source = null;
        try {
            typeName = ds.getTypeNames()[0];
            source = ds.getFeatureSource(typeName);
        } catch (IOException e) {
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
        // Filter on the data store by selecting only the geometries contained
        // into the input Geometry
        FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2(null);

        Filter filter = ff.within(ff.property(geometryPropertyName), ff.literal(geom));
        // Feature collection selection
        FeatureCollection coll = null;
        try {
            // coll = source.getFeatures();
            coll = source.getFeatures(filter);
        } catch (IOException e) {
            throw new ProcessException(e);
        }

        if (coll == null) {
            throw new ProcessException("Source Feature collection not found");
        }

        // Iterator on the features
        FeatureIterator iter = coll.features();

        double totalPerimeter = 0;

        // Selection of the inner list
        // List<Double> areas = new ArrayList<Double>();

        // double totalArea = 0;
        // Cycle on each polygon
        try {
            while (iter.hasNext()) {
                SimpleFeature feature = (SimpleFeature) iter.next();
                Geometry sourceGeometry = (Geometry) feature.getDefaultGeometry();
                // If the geometry is a Polygon, then the operations are
                // executed
                // reprojection of the polygon
                Geometry geoPrj = reprojectToEqualArea(sourceCRS, sourceGeometry);
                if (geoPrj != null) {
                    // Area/Perimeter calculation
                    // double area = geoPrj.getArea();
                    // areas.add(area);
                    // totalArea += area;
                    totalPerimeter += geoPrj.getLength();
                }
            }
        } catch (Exception e) {
            throw new ProcessException(e);
        } finally {
            // Iterator closure
            iter.close();
        }
        // Datastore dispose
        ds.dispose();

        if (reference) {
            // areasRef = areas;
            // totalAreaRef = totalArea;
            totalPerimeterRef = totalPerimeter;
        } else {
            // areasCur = areas;
            // totalAreaCur = totalArea;
            totalPerimeterCur = totalPerimeter;
        }
    }

    private static Geometry reprojectToEqualArea(CoordinateReferenceSystem sourceCRS,
            Geometry sourceGeometry) throws FactoryException, TransformException {
        // Reproject to the Lambert Equal Area

        final ReferencedEnvelope env = new ReferencedEnvelope(sourceGeometry.getEnvelopeInternal(),
                sourceCRS);
        // Geometry center used for centering the reprojection on the
        // Geometry(reduces distance artifacts)
        double lat = env.getMedian(1);
        double lon = env.getMedian(0);
        // Geometry center used for centering the reprojection on the
        // Geometry(reduces distance artifacts)
        Point center = sourceGeometry.getCentroid();
        // Creation of the MathTransform associated to the reprojection
        MathTransform transPoint = CRS.findMathTransform(sourceCRS, CRS.decode("EPSG:4326"), true);
        Point centerRP = (Point) JTS.transform(center, transPoint);
        lon = centerRP.getY();
        lat = centerRP.getX();
        // Creation of a wkt for the selected Geometry
        String wkt = UrbanGridProcess.PROJ_4326.replace("%LAT0%", String.valueOf(lat));
        wkt = wkt.replace("%LON0%", String.valueOf(lon));
        // Parsing of the selected WKT
        final CoordinateReferenceSystem targetCRS = CRS.parseWKT(wkt);
        // Creation of the MathTransform associated to the reprojection
        MathTransform trans = CRS.findMathTransform(sourceCRS, targetCRS, true);
        // Geometry reprojection
        Geometry geoPrj;
        if (!trans.isIdentity()) {
            geoPrj = JTS.transform(sourceGeometry, trans);
        } else {
            geoPrj = sourceGeometry;
        }
        return geoPrj;
    }
}
