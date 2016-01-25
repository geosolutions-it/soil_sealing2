/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import it.geosolutions.jaiext.changematrix.ChangeMatrixDescriptor.ChangeMatrix;

import java.util.Set;
import java.util.TreeSet;

/**
 * This is the changeMatrix
 * 
 * @author Damiano Giampaoli, GeoSolutions SAS
 * 
 */
public class ChangeMatrixDTO {

    private String rasterName = null;
    private String refYear = null;
    private String nowYear = null;

    /**
     * The implementation of the changeMatrix as a Set
     */
    private final Set<ChangeMatrixElement> changeMatrix = new TreeSet<ChangeMatrixElement>();

    /**
     * Init the changeMatrix as an empty TreeSet
     * 
     * @param classes
     * @param cm
     */
    public ChangeMatrixDTO(ChangeMatrix cm, Set<Integer> classes, String rasterName, String refYear, String nowYear, Double multiplier) {
        
        double value = 1;
        
        if(multiplier!=null){
            value = multiplier;
        }
        
        this.setRasterName(rasterName);
        if (refYear != null) this.setRefYear(refYear);
        if (nowYear != null) this.setNowYear(nowYear);

        for (Integer elRef : classes) {
            for (Integer elNow : classes) {
                final ChangeMatrixElement cme = new ChangeMatrixElement(elRef, elNow,
                        cm.retrievePairOccurrences(elRef, elNow) * value, cm.retrieveTotalArea(
                                elRef, elNow));
                add(cme);
            }
        }
    }

    /**
     * Init the changeMatrix as an empty TreeSet
     * 
     * @param classes
     * @param cm
     */
    public ChangeMatrixDTO(ChangeMatrix cm, Set<Integer> classes, String rasterName, String refYear, String nowYear) {
        this(cm, classes, rasterName, refYear, nowYear, null);
    }

    /**
     * Default constructor.
     * 
     */
    public ChangeMatrixDTO() {
    }

    /**
     * Add an element to the changeMatrix
     * 
     * @param el
     */
    public void add(ChangeMatrixElement el) {
        this.changeMatrix.add(el);
    }

    /**
     * 
     * @return
     */
    public Set<ChangeMatrixElement> getChangeMatrix() {
        return changeMatrix;
    }

    /**
     * @param rasterName the rasterName to set
     */
    public void setRasterName(String rasterName) {
        this.rasterName = rasterName;
    }

    /**
     * @return the rasterName
     */
    public String getRasterName() {
        return rasterName;
    }

    /**
     * @return the refYear
     */
    public String getRefYear() {
        return refYear;
    }

    /**
     * @param refYear the refYear to set
     */
    public void setRefYear(String refYear) {
        this.refYear = refYear;
    }

    /**
     * @return the nowYear
     */
    public String getNowYear() {
        return nowYear;
    }

    /**
     * @param nowYear the nowYear to set
     */
    public void setNowYear(String nowYear) {
        this.nowYear = nowYear;
    }

}
