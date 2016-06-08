/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.SampleModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import javax.media.jai.TiledImage;

import org.geoserver.wps.gs.soilsealing.CLCProcess.StatisticContainer;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridCoverageFactory;
import org.geotools.factory.GeoTools;
import org.geotools.geometry.Envelope2D;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opengis.geometry.Envelope;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LinearRing;
import com.vividsolutions.jts.geom.Polygon;

public class CLCProcessTest {

    public static final int DEF_H = 256;

    public static final int DEF_W = 256;

    public static final int DEF_TILE_H = 32;

    public static final int DEF_TILE_W = 32;

    public static final double AREA = CLCProcess.PIXEL_AREA;

    public static final int DEFAULT_POP_REF = 10;

    public static final int DEFAULT_POP_NOW = 20;

    public static final double DELTA = 0.0001;

    private static GridCoverage2D reference;

    private static GridCoverage2D now;

    private static List<Geometry> rois;

    private static Set<Integer> classes;

    private static List<List<Integer>> populations;

    private static double roiArea;

    private static double clc3test1;

    private static double clc4test1;

    private static double clc3test2;

    private static double clc4test2;

    private static double clc3test3;

    private static double clc4test3;

    private static double clctest4;

    private static double clctest5;

    public static GridCoverage2D createImage(boolean reference) {

        SampleModel sm = new ComponentSampleModel(DataBuffer.TYPE_BYTE, DEF_W, DEF_H, 1, DEF_W,
                new int[] { 0 });

        TiledImage img = new TiledImage(sm, DEF_TILE_W, DEF_TILE_H);

        int minX = 0;
        int maxX = 0;
        int minY = 0;
        int maxY = 0;

        int value1 = 3;
        int value2 = 4;

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

        int threshold = minY + (maxY - minY) / 2 - 1;

        for (int i = minX; i < maxX; i++) {
            for (int j = minY; j < maxY; j++) {

                if (j > threshold) {
                    img.setSample(i, j, 0, value2);
                } else {
                    img.setSample(i, j, 0, value1);
                }
            }
        }
        Envelope envelope = new Envelope2D(null, 0, 0, DEF_W, DEF_H);

        GridCoverage2D result = new GridCoverageFactory(GeoTools.getDefaultHints()).create("test",
                img, envelope);

        return result;
    }

    @BeforeClass
    public static void setup() {
        reference = createImage(true);
        now = createImage(false);

        rois = new ArrayList<Geometry>(1);

        GeometryFactory fact = new GeometryFactory();
        Coordinate[] coordinates = new Coordinate[5];
        for (int i = 0; i < coordinates.length; i++) {
            // if(i == 0 || i == 4){
            // coordinates[i] = new Coordinate(0-0.5, 0-0.5);
            // }else if(i == 3){
            // coordinates[i] = new Coordinate(DEF_W/2+0.5 , 0-0.5);
            // }else if(i == 1){
            // coordinates[i] = new Coordinate(0-0.5, DEF_H/2+0.5);
            // }else{
            // coordinates[i] = new Coordinate(DEF_W/2+0.5, DEF_H/2+0.5);
            // }
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
        rois.add(poly);

        classes = new TreeSet<Integer>();
        classes.add(Integer.valueOf(3));
        classes.add(Integer.valueOf(4));

        // Pops

        populations = new ArrayList<List<Integer>>(2);
        List<Integer> popref = new ArrayList<Integer>(1);
        popref.add(DEFAULT_POP_REF);
        populations.add(popref);

        List<Integer> popnow = new ArrayList<Integer>(1);
        popnow.add(DEFAULT_POP_NOW);
        populations.add(popnow);

        // PopVariation
        double deltaPop = DEFAULT_POP_NOW - DEFAULT_POP_REF;

        double correctArea = AREA * UrbanGridProcess.HACONVERTER;

        // Expected result init
        roiArea = DEF_H * DEF_W / 4 * correctArea;

        double class3Exp1 = 50 * correctArea;
        double class4Exp1 = 50 * correctArea;

        double class3Exp1test2 = 45 * correctArea;
        double class4Exp1test2 = 45 * correctArea;

        // Test 1 values
        clc3test1 = class3Exp1 / roiArea;
        clc4test1 = class4Exp1 / roiArea;
        // Test 2 values
        clc3test2 = class3Exp1test2 / roiArea;
        clc4test2 = class4Exp1test2 / roiArea;
        // Test 3 values
        clc3test3 = (class3Exp1test2 - class3Exp1) / class3Exp1 * 100;
        clc4test3 = (class4Exp1test2 - class4Exp1) / class4Exp1 * 100;
        // Test 4 values
        double sumCLCref = class3Exp1 + class4Exp1;
        double sumCLCnow = class3Exp1test2 + class4Exp1test2;

        double deltaArea = sumCLCnow - sumCLCref;

        // Index result
        // if (deltaArea >= 0) {
        // if (deltaPop >= 0) {
        // clctest4 = IndexColor.GREEN.getValue();
        // } else {
        // clctest4 = IndexColor.YELLOW.getValue();
        // }
        // } else {
        // if (deltaPop >= 0) {
        // clctest4 = IndexColor.RED.getValue();
        // } else {
        // clctest4 = IndexColor.BLUE.getValue();
        // }
        // }
        clctest4 = deltaArea / deltaPop;

        // Test 5 values
        double sprawl = (deltaArea / sumCLCref) / (deltaPop / DEFAULT_POP_REF);

        // if (sprawl > CLCProcess.UPPER_BOUND_INDEX_4) {
        // clctest5 = IndexColor.RED.getValue();
        // } else if (sprawl < CLCProcess.LOWER_BOUND_INDEX_4) {
        // clctest5 = IndexColor.GREEN.getValue();
        // } else {
        // clctest5 = IndexColor.YELLOW.getValue();
        // }

        clctest5 = sprawl;
    }

    @Test
    public void testIdx1() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, null, classes,
                SoilSealingIndexType.COVERAGE_COEFFICIENT, AREA, rois, populations, null, null);
        StatisticContainer container = result.get(0);

        double class3 = container.getResultsRef()[0];
        double class4 = container.getResultsRef()[1];

        Assert.assertEquals(class3, clc3test1, DELTA);
        Assert.assertEquals(class4, clc4test1, DELTA);
    }

    @Test
    public void testIdx1With2Img() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.COVERAGE_COEFFICIENT, AREA, rois, populations, null, null);
        StatisticContainer container = result.get(0);

        double class3r = container.getResultsRef()[0];
        double class4r = container.getResultsRef()[1];

        double class3n = container.getResultsNow()[0];
        double class4n = container.getResultsNow()[1];

        Assert.assertEquals(class3r, clc3test1, DELTA);
        Assert.assertEquals(class4r, clc4test1, DELTA);

        Assert.assertEquals(class3n, clc3test2, DELTA);
        Assert.assertEquals(class4n, clc4test2, DELTA);
    }

    @Test
    public void testIdx2() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.RATE_OF_CHANGE, AREA, rois, populations, null, null);

        StatisticContainer container = result.get(0);

        double class3 = container.getResults()[0];
        double class4 = container.getResults()[1];

        Assert.assertEquals(class3, clc3test3, DELTA);
        Assert.assertEquals(class4, clc4test3, DELTA);
    }

    @Test
    public void testIdx3() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.MARGINAL_LAND_TAKE, AREA, rois, populations, null, null);

        StatisticContainer container = result.get(0);

        double idx = container.getResults()[0];

        Assert.assertEquals(idx, clctest4, DELTA);

    }

    @Test
    public void testIdx4() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.URBAN_SPRAWL, AREA, rois, populations, null, null);

        StatisticContainer container = result.get(0);

        double idx = container.getResults()[0];

        Assert.assertEquals(idx, clctest5, DELTA);
    }

    // EXCEPTION TESTS

    @Test(expected = IllegalArgumentException.class)
    public void testNoCoverages() {
        List<StatisticContainer> result = new CLCProcess().execute(null, null, classes,
                SoilSealingIndexType.COVERAGE_COEFFICIENT, AREA, rois, populations, null, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOnlyOneCoverage() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, null, classes,
                SoilSealingIndexType.URBAN_SPRAWL, AREA, rois, populations, null, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongIndex() {
        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.EDGE_DENSITY, AREA, rois, populations, null, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongPopulation() {
        List<List<Integer>> list = new ArrayList<List<Integer>>();

        list.add(populations.get(0));

        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.URBAN_SPRAWL, AREA, rois, list, null, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongPopulationForGeometry() {
        List<List<Integer>> list = new ArrayList<List<Integer>>();

        list.add(populations.get(0));
        list.add(new ArrayList<Integer>());

        List<StatisticContainer> result = new CLCProcess().execute(reference, now, classes,
                SoilSealingIndexType.URBAN_SPRAWL, AREA, rois, list, null, null);
    }

    @AfterClass
    public static void finalDispose() {
        reference.dispose(true);
        now.dispose(true);
    }

}
