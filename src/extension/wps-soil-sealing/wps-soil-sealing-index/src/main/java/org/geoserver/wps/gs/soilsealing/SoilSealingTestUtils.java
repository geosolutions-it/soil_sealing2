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

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.media.jai.DataBufferDouble;

import org.apache.commons.io.IOUtils;
import org.geoserver.wps.gs.CoverageImporter;
import org.geoserver.wps.gs.soilsealing.UrbanGridCUDAProcess.CUDABean;
import org.geotools.coverage.Category;
import org.geotools.coverage.GridSampleDimension;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridCoverageFactory;
import org.geotools.coverage.grid.io.AbstractGridCoverageWriter;
import org.geotools.coverage.grid.io.AbstractGridFormat;
import org.geotools.data.DataUtilities;
import org.geotools.data.collection.ListFeatureCollection;
import org.geotools.data.shapefile.shp.ShapefileWriter;
import org.geotools.data.simple.SimpleFeatureCollection;
import org.geotools.factory.GeoTools;
import org.geotools.factory.Hints;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.gce.arcgrid.ArcGridFormat;
import org.geotools.gce.arcgrid.ArcGridWriter;
import org.geotools.gce.geotiff.GeoTiffFormat;
import org.geotools.gce.geotiff.GeoTiffWriter;
import org.geotools.referencing.CRS;
import org.opengis.coverage.grid.GridCoverage;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.geometry.Envelope;
import org.opengis.parameter.GeneralParameterValue;
import org.opengis.parameter.ParameterValueGroup;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.sun.media.imageioimpl.common.ImageUtil;
import com.vividsolutions.jts.geom.Geometry;

/**
 * @author geosolutions
 *
 */
public class SoilSealingTestUtils {

    /** Set whether the superuser is performing a test a (cuda) codes to calculate GUI indices */
    public final static boolean TESTING = true;

    public final static String TESTING_DIR = "/media/DATI/db-backup/ssgci-data/testing";// "/opt/soil_sealing/exchange_data/testing";

    public static void storeGeoTIFFSampleImage(List<CUDABean> beans, int w, int h, Object data,
            int dataType, String name) throws IOException {
        storeGeoTIFFSampleImage(
                beans.get(0).getReferenceCoverage().getEnvelope(),
                beans.get(0).getReferenceCoverage(),
                w, h, data, dataType, name);
    }
    
    @SuppressWarnings("serial")
    public static void storeGeoTIFFSampleImage(
            Envelope envelope, 
            GridCoverage coverage, 
            int w, int h, 
            Object data,
            int dataType, String name) throws IOException {
        /**
         * create the final coverage using final envelope
         */
        // Definition of the SampleModel
        final SampleModel sm = new PixelInterleavedSampleModel(dataType, w, h, 1, w,
                new int[] { 0 });
        // DataBuffer containing input data

        DataBuffer db1 = null;
        if (dataType == DataBuffer.TYPE_INT) {
            db1 = new DataBufferInt((int[]) data, w * h);
        } else if (dataType == DataBuffer.TYPE_BYTE) {
            db1 = new DataBufferByte((byte[]) data, w * h);
        } else if (dataType == DataBuffer.TYPE_DOUBLE) {
            db1 = new DataBufferDouble((double[]) data, w * h);
        }
        // Writable Raster used for creating the BufferedImage
        final WritableRaster wr = com.sun.media.jai.codecimpl.util.RasterFactory
                .createWritableRaster(sm, db1, new Point(0, 0));
        final BufferedImage image = new BufferedImage(ImageUtil.createColorModel(sm), wr, false,
                null);

        // hints for tiling
        final Hints hints = GeoTools.getDefaultHints().clone();

        // build the output sample dimensions, use the default value ( 0 ) as
        // the no data
        final GridSampleDimension outSampleDimension = new GridSampleDimension("classification");

        final GridCoverage2D retValue = new GridCoverageFactory(hints).create(name, image,
                envelope,
                new GridSampleDimension[] { outSampleDimension },
                new GridCoverage[] { coverage },
                null);

        final File file = new File(SoilSealingTestUtils.TESTING_DIR,
                name/* +(System.nanoTime()) */ + ".tif");
        AbstractGridCoverageWriter writer = new GeoTiffWriter(file) ;//GeoTiffWriter(file);

        // setting the write parameters for this geotiff
        final ParameterValueGroup gtiffParams = new ArcGridFormat().getWriteParameters();
        gtiffParams.parameter(ArcGridFormat.GEOTOOLS_WRITE_PARAMS.getName().toString())
                .setValue(CoverageImporter.DEFAULT_WRITE_PARAMS);
        final GeneralParameterValue[] wps = (GeneralParameterValue[]) gtiffParams.values()
                .toArray(new GeneralParameterValue[1]);

        try {
            writer.write(retValue, wps);
        } finally {
            try {
                writer.dispose();
            } catch (Exception e) {
                throw new IOException("Unable to write the output raster.", e);
            }
        }
    }
    
    public static void storeGeometryAsWKT(Geometry geom, String shpBaseFileName, CoordinateReferenceSystem coordinateReferenceSystem) {
        FileOutputStream output;
        FileOutputStream prj;
        try {
            output = new FileOutputStream(new File(TESTING_DIR, shpBaseFileName+".wkt"));
            IOUtils.write(geom.toText(), output);
            
            CoordinateReferenceSystem crs = (geom.getSRID()!=0?CRS.decode("EPSG:"+geom.getSRID()):coordinateReferenceSystem);
            prj = new FileOutputStream(new File(TESTING_DIR, shpBaseFileName+".prj"));
            IOUtils.write(crs.toWKT(), prj);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    
    @SuppressWarnings("resource")
    public static void storeGeometryAsShapeFile(Geometry geom, String shpBaseFileName) {
        FileChannel shpChannel = null;
        FileChannel shxChannel = null;
        try {
            shpChannel = new FileOutputStream(new File(TESTING_DIR, shpBaseFileName+".shp")).getChannel();
            shxChannel = new FileOutputStream(new File(TESTING_DIR, shpBaseFileName+".shx")).getChannel();
            ShapefileWriter writer = null;
            try {
                writer = new ShapefileWriter(shpChannel, shxChannel);
                
                SimpleFeatureType schema = DataUtilities.createType(
                        "", 
                        "Location", 
                        "locations:Polygon:srid="+geom.getSRID()+"," + // <- the geometry // attribute:
                        "id:Integer" // a number attribute
                         //add more attributes here to match your dataset
                    );
                SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(schema);
                //for each feature you have do:
                featureBuilder.add(geom);
                featureBuilder.add(shpBaseFileName);
                featureBuilder.add(1);
                //+ all the other attributes (probably use a loop)
                SimpleFeature feature = featureBuilder.buildFeature(null);
                
                //put them into a collection feats and then convert it to a FeatureCollection
                List<SimpleFeature> feats = new ArrayList<SimpleFeature>();
                feats.add(feature);
                SimpleFeatureCollection collection = new ListFeatureCollection(schema, feats);
                //writer.writeHeaders(arg0, arg1, arg2, arg3);
                writer.skipHeaders();
                writer.writeGeometry(geom);
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            } finally {
                if (writer != null) {
                    try {
                        writer.close();
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
            }
        } catch(Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            if (shpChannel != null && shpChannel.isOpen()) {
                try {
                    shpChannel.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            
            if (shxChannel != null && shxChannel.isOpen()) {
                try {
                    shxChannel.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
    }
}
