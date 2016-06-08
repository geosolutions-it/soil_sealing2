/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing.model;

import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingIndexType;
import org.geoserver.wps.gs.soilsealing.SoilSealingImperviousnessProcess.SoilSealingSubIndexType;

public class SoilSealingIndex {

    private SoilSealingIndexType id;

    private SoilSealingSubIndexType subindex;

    public SoilSealingIndex(SoilSealingIndexType soilSealingIndexType, SoilSealingSubIndexType subindex) {
        super();
        this.id = soilSealingIndexType;
        this.subindex = subindex;
    }

    /**
     * @return the id
     */
    public SoilSealingIndexType getId() {
        return id;
    }

    /**
     * @param id the id to set
     */
    public void setId(SoilSealingIndexType id) {
        this.id = id;
    }

    /**
     * @return the subindex
     */
    public SoilSealingSubIndexType getSubindex() {
        return subindex;
    }

    /**
     * @param subindex the subindex to set
     */
    public void setSubindex(SoilSealingSubIndexType subindex) {
        this.subindex = subindex;
    }

}
