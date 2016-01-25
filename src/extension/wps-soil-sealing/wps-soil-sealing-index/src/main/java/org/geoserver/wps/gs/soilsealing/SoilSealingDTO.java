/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import org.geoserver.wps.gs.soilsealing.model.SoilSealingIndex;
import org.geoserver.wps.gs.soilsealing.model.SoilSealingTime;

public class SoilSealingDTO {

    private SoilSealingIndex index;
    private SoilSealingTime curTime;
    private SoilSealingTime refTime;
    private String diffImageName;
    
    /**
     * Default Constructor
     */
    public SoilSealingDTO() {

    }

    /**
     * @return the index
     */
    public SoilSealingIndex getIndex() {
        return index;
    }

    /**
     * @param index the index to set
     */
    public void setIndex(SoilSealingIndex index) {
        this.index = index;
    }

    /**
     * @return the curTime
     */
    public SoilSealingTime getCurTime() {
        return curTime;
    }

    /**
     * @param curTime the curTime to set
     */
    public void setCurTime(SoilSealingTime curTime) {
        this.curTime = curTime;
    }

    /**
     * @return the refTime
     */
    public SoilSealingTime getRefTime() {
        return refTime;
    }

    /**
     * @param refTime the refTime to set
     */
    public void setRefTime(SoilSealingTime refTime) {
        this.refTime = refTime;
    }

    public String getDiffImageName() {
        return diffImageName;
    }

    public void setDiffImageName(String diffImageName) {
        this.diffImageName = diffImageName;
    }
    
    
}
