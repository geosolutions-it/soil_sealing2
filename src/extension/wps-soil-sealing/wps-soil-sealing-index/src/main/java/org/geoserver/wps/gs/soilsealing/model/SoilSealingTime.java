/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing.model;

public class SoilSealingTime {

    private String time;

    private SoilSealingOutput output;

    /**
     * @param time
     * @param output
     */
    public SoilSealingTime(String time, SoilSealingOutput output) {
        super();
        this.time = time;
        this.output = output;
    }

    /**
     * @return the time
     */
    public String getTime() {
        return time;
    }

    /**
     * @param time the time to set
     */
    public void setTime(String time) {
        this.time = time;
    }

    /**
     * @return the output
     */
    public SoilSealingOutput getOutput() {
        return output;
    }

    /**
     * @param output the output to set
     */
    public void setOutput(SoilSealingOutput output) {
        this.output = output;
    }

}
