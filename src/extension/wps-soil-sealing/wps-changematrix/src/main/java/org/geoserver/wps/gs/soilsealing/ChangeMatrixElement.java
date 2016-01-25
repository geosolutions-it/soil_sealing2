/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

/**
 * This Bean represent a single element of a ChangeMatrix
 * 
 * @author DamianoG, GeoSolutions
 * 
 */
public class ChangeMatrixElement implements Comparable<ChangeMatrixElement> {

    /**
     * The Pixel class type, Image at time 0
     */
    private int ref;

    /**
     * The Pixel class type, Image at time 1
     */
    private int now;

    /**
     * The number of pixel that has been changed from class ref to class now
     */
    private double pixels;

    /**
     * The total area of the pixel that has been changed from class ref to class now
     */
    private double area;

    /**
     * @param ref
     * @param now
     * @param pixels
     */
    public ChangeMatrixElement(int ref, int now, double pixels) {
        this(ref, now, pixels, 0);
    }

    /**
     * @param ref
     * @param now
     * @param pixels
     * @param area
     */
    public ChangeMatrixElement(int ref, int now, double pixels, double area) {
        this.ref = ref;
        this.now = now;
        this.pixels = pixels;
        this.area = area;
    }

    /**
     * @return the ref
     */
    public int getRef() {
        return ref;
    }

    /**
     * @return the now
     */
    public int getNow() {
        return now;
    }

    /**
     * @return the pixels
     */
    public double getPixels() {
        return pixels;
    }

    /**
     * @return the total area
     */
    public double getArea() {
        return area;
    }

    /**
     * @param ref the ref to set
     */
    public void setRef(int ref) {
        this.ref = ref;
    }

    /**
     * @param now the now to set
     */
    public void setNow(int now) {
        this.now = now;
    }

    /**
     * @param pixels the pixels to set
     */
    public void setPixels(int pixels) {
        this.pixels = pixels;
    }

    /**
     * @param area the total area to set
     */
    public void setArea(int area) {
        this.area = area;
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Comparable#compareTo(java.lang.Object)
     */
    @Override
    public int compareTo(ChangeMatrixElement o) {
        double area1 = this.getArea();
        double area2 = o.getArea();
        double perc = (area1 - area2)/area1;
        if (o.getRef() == this.getRef() && o.getNow() == this.getNow()
                && o.getPixels() == this.getPixels() && perc < 0.01) {
            return 0;
        }
        return 1;
    }
}
