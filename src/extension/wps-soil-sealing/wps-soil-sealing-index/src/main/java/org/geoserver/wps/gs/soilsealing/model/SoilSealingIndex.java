/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing.model;


public class SoilSealingIndex {

    private int id;

    private String subindex;

    public SoilSealingIndex(int id, String subindex) {
        super();
        this.id = id;
        this.subindex = subindex;
    }

    /**
     * @return the id
     */
    public int getId() {
        return id;
    }

    /**
     * @param id the id to set
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * @return the subindex
     */
    public String getSubindex() {
        return subindex;
    }

    /**
     * @param subindex the subindex to set
     */
    public void setSubindex(String subindex) {
        this.subindex = subindex;
    }

}
