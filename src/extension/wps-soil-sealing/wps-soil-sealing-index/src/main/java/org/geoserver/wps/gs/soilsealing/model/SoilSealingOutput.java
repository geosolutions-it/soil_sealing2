/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing.model;

public class SoilSealingOutput {
    private String referenceName;
    
    private String layerName;

    private String[] admUnits;

    private String[] clcLevels;

    private double[][] values;

    /**
     * @param referenceName
     * @param admUnits
     * @param clcLevels
     * @param values
     */
    public SoilSealingOutput(String referenceName, String[] admUnits, String[] clcLevels, double[][] values) {
        super();
        this.referenceName = referenceName;
        this.admUnits = admUnits;
        this.clcLevels = clcLevels;
        this.values = values;
    }

    /**
     * @return the referenceName
     */
    public String getReferenceName() {
        return referenceName;
    }

    /**
     * @param referenceName the referenceName to set
     */
    public void setReferenceName(String referenceName) {
        this.referenceName = referenceName;
    }

    /**
     * @return the layerName
     */
    public String getLayerName() {
        return layerName;
    }

    /**
     * @param layerName the layerName to set
     */
    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    /**
     * @return the admUnits
     */
    public String[] getAdmUnits() {
        return admUnits;
    }

    /**
     * @param admUnits the admUnits to set
     */
    public void setAdmUnits(String[] admUnits) {
        this.admUnits = admUnits;
    }

    /**
     * @return the clcLevels
     */
    public String[] getClcLevels() {
        return clcLevels;
    }

    /**
     * @param clcLevels the clcLevels to set
     */
    public void setClcLevels(String[] clcLevels) {
        this.clcLevels = clcLevels;
    }

    /**
     * @return the values
     */
    public double[][] getValues() {
        return values;
    }

    /**
     * @param values the values to set
     */
    public void setValues(double[][] values) {
        this.values = values;
    }

}
