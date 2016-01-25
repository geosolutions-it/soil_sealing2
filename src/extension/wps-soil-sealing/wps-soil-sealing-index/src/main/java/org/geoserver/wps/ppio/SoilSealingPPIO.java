/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.ppio;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;

import net.sf.json.JSONSerializer;

import org.apache.commons.io.IOUtils;
import org.geoserver.wps.gs.soilsealing.SoilSealingDTO;

/**
 * @author Alessio Fabiani, GeoSolutions SAS
 * 
 */
public class SoilSealingPPIO extends CDataPPIO {

    public SoilSealingPPIO() {
        super(SoilSealingDTO.class, SoilSealingDTO.class, "application/json");
    }

    @Override
    public Object decode(InputStream input) throws Exception {
        return null;
    }

    @Override
    public void encode(Object value, OutputStream os) throws Exception {
        PrintWriter pw = new PrintWriter(os);
        try {

            // pw.write(JSONSerializer.toJSON(((ChangeMatrixDTO)value).getChangeMatrix()).toString());
            pw.write(JSONSerializer.toJSON(((SoilSealingDTO) value)).toString());
        } finally {
            IOUtils.closeQuietly(pw);
            IOUtils.closeQuietly(os);
        }
    }

    @Override
    public String getFileExtension() {
        return "json";
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.geoserver.wps.ppio.CDataPPIO#decode(java.lang.String)
     */
    @Override
    public Object decode(String input) throws Exception {
        return null;
    }

}
