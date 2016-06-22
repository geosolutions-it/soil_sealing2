/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.geoserver.catalog.FeatureTypeInfo;
import org.geotools.data.DefaultTransaction;
import org.geotools.data.FeatureReader;
import org.geotools.data.Query;
import org.geotools.data.Transaction;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.geometry.jts.JTS;
import org.geotools.jdbc.JDBCDataStore;
import org.geotools.referencing.CRS;
import org.geotools.referencing.operation.transform.AffineTransform2D;
import org.geotools.referencing.operation.transform.ProjectiveTransform;
import org.geotools.util.SoftValueHashMap;
import org.geotools.util.logging.Logging;
import org.opengis.feature.Feature;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.PrecisionModel;
import com.vividsolutions.jts.simplify.DouglasPeuckerSimplifier;

/**
 * @author GeoSolutions
 *
 */
public class SoilSealingProcessingUtils {

    private final static Logger LOGGER = Logging.getLogger(SoilSealingProcessingUtils.class);
    
    public static final int DEGREES_TO_METER_RATIO = 111128;

    /**
     * Geometry and Filter Factories
     */
    public static final FilterFactory ff = CommonFactoryFinder.getFilterFactory(null);

    public static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory(new PrecisionModel());
    
    /**
     * Caches the wbodies Geometries grabbed from the DataStore
     */
    static SoftValueHashMap<String, List<Geometry>> waterBodiesMaskCache = new SoftValueHashMap<String, List<Geometry>>(2);
    
    static SoftValueHashMap<String, Geometry> masks;

    
    public SoilSealingProcessingUtils() {
        SoilSealingProcessingUtils.waterBodiesMaskCache.clear();
    }
    
    /**
     * @param waterBodiesMaskReference
     * @param mask
     * @return 
     * @throws IOException
     */
    public synchronized static Geometry getWBodiesMask(FeatureTypeInfo waterBodiesMaskReference)
            throws IOException {
        
        if (!SoilSealingProcessingUtils.waterBodiesMaskCache.containsKey(waterBodiesMaskReference.getName()) ||
                SoilSealingProcessingUtils.waterBodiesMaskCache.get(waterBodiesMaskReference.getName()).isEmpty()) {        
            FeatureReader<SimpleFeatureType, SimpleFeature> ftReader = null;
            Transaction transaction = new DefaultTransaction();
            try {
                final JDBCDataStore ds = (JDBCDataStore) waterBodiesMaskReference.getStore().getDataStore(null);
                
                Query query = new Query(waterBodiesMaskReference.getFeatureType().getName().getLocalPart(), Filter.INCLUDE);
                
                ftReader = ds.getFeatureReader(query, transaction);
                
                List<Geometry> theGeoms = new LinkedList<Geometry>();
                while (ftReader.hasNext()) {
                    Feature feature = ftReader.next();
                    final Geometry theGeom = (Geometry) feature.getDefaultGeometryProperty().getValue();
                    if (!masks.containsKey(waterBodiesMaskReference.getName()) || masks.get(waterBodiesMaskReference.getName()) == null) {
                        masks.put(waterBodiesMaskReference.getName(), theGeom);
                    } else {
                        Geometry mask = masks.get(waterBodiesMaskReference.getName());
                        masks.put(waterBodiesMaskReference.getName(), mask.union(theGeom));
                    }
                    theGeoms.add(theGeom);
                }
                SoilSealingProcessingUtils.waterBodiesMaskCache.put(waterBodiesMaskReference.getName(), theGeoms);
            } catch (Exception e) {
                LOGGER.log(Level.WARNING, "Error while getting Water Bodies Mask Geometries.", e);
                masks.remove(waterBodiesMaskReference.getName());
            } finally {
                if (ftReader != null) {
                    ftReader.close();
                }
    
                transaction.commit();
                transaction.close();
            }
        } else {
            if (!masks.containsKey(waterBodiesMaskReference.getName()) || masks.get(waterBodiesMaskReference.getName()) == null) {
                for (Geometry theGeom : SoilSealingProcessingUtils.waterBodiesMaskCache.get(waterBodiesMaskReference.getName())) {
                    if (!masks.containsKey(waterBodiesMaskReference.getName()) || masks.get(waterBodiesMaskReference.getName()) == null) {
                        masks.put(waterBodiesMaskReference.getName(), theGeom);
                    } else {
                        Geometry mask = masks.get(waterBodiesMaskReference.getName());
                        masks.put(waterBodiesMaskReference.getName(), mask.union(theGeom));
                    }                    
                }
            }
        }
        
        return (Geometry) masks.get(waterBodiesMaskReference.getName()).clone();
    }

    /**
     * @return the mask
     */
    public Geometry getMask(String waterBodiesMaskReference) {
        return (Geometry) masks.get(waterBodiesMaskReference).clone();
    }
    
    /**
     * 
     * @param theGeom
     * @param referenceCrs
     * @param gridToWorldCorner
     * @return
     * @throws NoSuchAuthorityCodeException
     * @throws FactoryException
     * @throws MismatchedDimensionException
     * @throws TransformException
     * @throws NoninvertibleTransformException
     */
    public static Geometry toReferenceCRS(Geometry theGeom, CoordinateReferenceSystem referenceCrs,
            AffineTransform gridToWorldCorner, boolean toRasterSpace)
            throws NoSuchAuthorityCodeException, FactoryException, MismatchedDimensionException,
            TransformException, NoninvertibleTransformException {
        // check if we need to reproject the ROI from WGS84 (standard in the input) to the reference CRS
        if (theGeom.getSRID() <= 0)
            theGeom.setSRID(CRS.lookupEpsgCode(referenceCrs, true));
        final CoordinateReferenceSystem targetCrs = CRS.decode("EPSG:" + theGeom.getSRID(), true);
        if (CRS.equalsIgnoreMetadata(referenceCrs, targetCrs)) {
            Geometry rasterSpaceGeometry = JTS.transform(theGeom,
                    new AffineTransform2D(gridToWorldCorner.createInverse()));
            return (toRasterSpace ? DouglasPeuckerSimplifier.simplify(rasterSpaceGeometry, 1)
                    : theGeom);
        } else {
            // reproject
            MathTransform transform = CRS.findMathTransform(targetCrs, referenceCrs, true);
            Geometry roiPrj;
            if (transform.isIdentity()) {
                roiPrj = theGeom;
                roiPrj.setSRID(CRS.lookupEpsgCode(targetCrs, true));
            } else {
                roiPrj = JTS.transform(theGeom, transform);
                roiPrj.setSRID(CRS.lookupEpsgCode(referenceCrs, true));
            }
            return (toRasterSpace
                    ? JTS.transform(roiPrj, ProjectiveTransform.create(gridToWorldCorner).inverse())
                    : roiPrj);
        }
    }
}
