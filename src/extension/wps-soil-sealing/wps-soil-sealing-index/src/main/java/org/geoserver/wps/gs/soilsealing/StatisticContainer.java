/*
 *    GeoTools - The Open Source Java GIS Toolkit
 *    http://geotools.org
 *
 *    (C) 2016, Open Source Geospatial Foundation (OSGeo)
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation;
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 */
package org.geoserver.wps.gs.soilsealing;

import java.awt.image.RenderedImage;

import com.vividsolutions.jts.geom.Geometry;

/**
 * @author geosolutions
 *
 */
/**
 * Helper class used for storing the index results for each Geometry and passing it in output
 */
public class StatisticContainer {

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
