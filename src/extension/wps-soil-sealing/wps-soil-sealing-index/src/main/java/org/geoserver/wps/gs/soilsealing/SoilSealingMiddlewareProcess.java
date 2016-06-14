package org.geoserver.wps.gs.soilsealing;

import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.geoserver.catalog.Catalog;
import org.geoserver.catalog.CoverageInfo;
import org.geoserver.catalog.FeatureTypeInfo;
import org.geoserver.config.GeoServer;
import org.geoserver.wps.WPSException;
import org.geoserver.wps.gs.soilsealing.SoilSealingAdministrativeUnit.AuSelectionType;
import org.geotools.coverage.grid.GridGeometry2D;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.geometry.Envelope2D;
import org.geotools.geometry.jts.JTS;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.process.gs.GSProcess;
import org.geotools.referencing.CRS;
import org.geotools.referencing.operation.transform.AffineTransform2D;
import org.geotools.referencing.operation.transform.ProjectiveTransform;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.metadata.spatial.PixelOrientation;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.datum.PixelInCell;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.PrecisionModel;
import com.vividsolutions.jts.simplify.DouglasPeuckerSimplifier;

@SuppressWarnings("deprecation")
public abstract class SoilSealingMiddlewareProcess implements GSProcess {

    public static final int DEGREES_TO_METER_RATIO = 111128;
    /**
     * Geometry and Filter Factories
     */
    protected static final FilterFactory ff = CommonFactoryFinder.getFilterFactory(null);
    protected static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory(new PrecisionModel());
    
    /**
     * The GeoServer Catalog
     */
    protected Catalog catalog;
    /**
     * The GeoServer Bean
     */
    protected GeoServer geoserver;

    /**
     * Default Constructor
     * 
     * @param catalog
     * @param geoserver
     */
    public SoilSealingMiddlewareProcess(Catalog catalog, GeoServer geoserver) {
        super();
        
        this.catalog = catalog;
        this.geoserver = geoserver;
    }

    /**
     * 
     * 
     * @param nowFilter
     * @param admUnits
     * @param admUnitSelectionType
     * @param ciReference
     * @param geoCodingReference
     * @param populationReference
     * @param municipalities
     * @param rois
     * @param populations
     * @param referenceYear
     * @param currentYear
     * @param referenceCrs 
     * @throws IOException
     * @throws NoSuchAuthorityCodeException
     * @throws FactoryException
     * @throws TransformException
     * @throws NoninvertibleTransformException
     */
    protected void prepareAdminROIs(Filter nowFilter, String admUnits,
            AuSelectionType admUnitSelectionType, CoverageInfo ciReference,
            FeatureTypeInfo geoCodingReference, FeatureTypeInfo populationReference,
            List<String> municipalities, List<Geometry> rois, List<List<Integer>> populations,
            final String referenceYear, final String currentYear, 
            CoordinateReferenceSystem referenceCrs, boolean toRasterSpace) throws IOException,
            NoSuchAuthorityCodeException, FactoryException, TransformException,
            NoninvertibleTransformException {
        // extract administrative units and geometries
        // //
        // GRID TO WORLD preparation from reference
        // //
        final AffineTransform gridToWorldCorner = (AffineTransform) ((GridGeometry2D) ciReference.getGrid()).getGridToCRS2D(PixelOrientation.UPPER_LEFT);
        
        for (String au : admUnits.split(",")) {
            SoilSealingAdministrativeUnit sAu = new SoilSealingAdministrativeUnit(au, geoCodingReference, populationReference);
            if (admUnitSelectionType == AuSelectionType.AU_LIST) {
                Geometry roi = null;
                int srID=0;
                int referencePopulation = 0;
                int currentPopulation = 0;
                switch (sAu.getType()) {
                    case MUNICIPALITY :
                        boolean hasPop = populateInputLists(nowFilter, referenceYear, currentYear,
                                gridToWorldCorner, referenceCrs, rois, populations, sAu, toRasterSpace);
                        if(hasPop){
                        	municipalities.add(sAu.getName() + " - " + sAu.getParent());
                        }
                        break;
                    case DISTRICT:
                        for(SoilSealingAdministrativeUnit ssAu : sAu.getSubs())
                        {
                            if (roi == null) {
                                    roi = toReferenceCRS(ssAu.getTheGeom(), referenceCrs, gridToWorldCorner, toRasterSpace);
                                    srID = roi.getSRID();
                                }
                            else roi = GEOMETRY_FACTORY.buildGeometry(Arrays.asList(roi, toReferenceCRS(ssAu.getTheGeom(), referenceCrs, gridToWorldCorner, toRasterSpace))).union();
                            if (ssAu.getPopulation() != null) {
                                if (ssAu.getPopulation().get(referenceYear) != null) referencePopulation += ssAu.getPopulation().get(referenceYear);
                                if (nowFilter != null && ssAu.getPopulation().get(currentYear) != null) currentPopulation += ssAu.getPopulation().get(currentYear);
                            }
                        }
                        roi.setSRID(srID);
                        rois.add(roi);
                        populations.get(0).add(referencePopulation);
                        if (nowFilter != null) populations.get(1).add(currentPopulation);
                        municipalities.add(sAu.getName() + " - " + sAu.getParent());
                        break;
                    case REGION:
                        for(SoilSealingAdministrativeUnit ssAu : sAu.getSubs())
                        {
                            for(SoilSealingAdministrativeUnit sssAu : ssAu.getSubs()) {
                                if (roi == null){
                                    roi = toReferenceCRS(sssAu.getTheGeom(), referenceCrs, gridToWorldCorner, toRasterSpace);
                                    srID = roi.getSRID();
                                }
                                else roi = GEOMETRY_FACTORY.buildGeometry(Arrays.asList(roi, toReferenceCRS(sssAu.getTheGeom(), referenceCrs, gridToWorldCorner, toRasterSpace))).union();
                                if (sssAu.getPopulation() != null) {
                                    if (sssAu.getPopulation().get(referenceYear) != null) referencePopulation += sssAu.getPopulation().get(referenceYear);
                                    if (nowFilter != null && sssAu.getPopulation().get(currentYear) != null) currentPopulation += sssAu.getPopulation().get(currentYear);
                                }
                            }
                        }
                        roi.setSRID(srID);
                        rois.add(roi);
                        populations.get(0).add(referencePopulation);
                        if (nowFilter != null) populations.get(1).add(currentPopulation);
                        municipalities.add(sAu.getName() + " - " + sAu.getParent());
                        break;
                    default:
                        break;
                }
            } else if (admUnitSelectionType == AuSelectionType.AU_SUBS) {
                switch (sAu.getType()) {
                    case MUNICIPALITY :
                        boolean hasPop = populateInputLists(nowFilter, referenceYear, currentYear, gridToWorldCorner, referenceCrs, rois, populations, sAu, toRasterSpace);
                        if(hasPop){
                        	municipalities.add(sAu.getName() + " - " + sAu.getParent());
                        }
                        //municipalities.add(sAu.getName() + " - " + sAu.getParent());
                        break;
                    case DISTRICT:
                        for(SoilSealingAdministrativeUnit ssAu : sAu.getSubs())
                        {
                            hasPop = populateInputLists(nowFilter, referenceYear, currentYear, gridToWorldCorner, referenceCrs, rois, populations, ssAu, toRasterSpace);
                            if(hasPop){
                            	municipalities.add(ssAu.getName() + " - " + ssAu.getParent());
                            }
                            //municipalities.add(ssAu.getName() + " - " + ssAu.getParent());
                        }
                        break;
                    case REGION:
                        for(SoilSealingAdministrativeUnit ssAu : sAu.getSubs())
                        {
                            for(SoilSealingAdministrativeUnit sssAu : ssAu.getSubs()) {
                                hasPop = populateInputLists(nowFilter, referenceYear, currentYear, gridToWorldCorner, referenceCrs, rois, populations, sssAu, toRasterSpace);
                                if(hasPop){
                                	municipalities.add(sssAu.getName() + " - " + sssAu.getParent());
                                }
                                //municipalities.add(sssAu.getName() + " - " + sssAu.getParent());
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
    
    /**
     * 
     * @param values
     * @return
     */
    static protected double[][] cleanUpValues(double[][] values) {
        if (values != null) {
            int i = 0;
            for (double[] d : values) {
                if (d != null) {
                    int j = 0;
                    for (double v : d) {
                        v = Double.isNaN(v) || Double.isInfinite(v) ? 0 : v;
                        values[i][j++] = v;
                    }
                }
                i++;
            }
        }
        
        return values;
    }

    /**
     * @param nowFilter
     * @param referenceYear
     * @param currentYear
     * @param gridToWorldCorner
     * @param referenceCrs
     * @param rois
     * @param populations
     * @param sAu
     * @param toRasterSpace 
     * @throws NoSuchAuthorityCodeException
     * @throws FactoryException
     * @throws TransformException
     * @throws NoninvertibleTransformException
     */
    protected boolean populateInputLists(Filter nowFilter, final String referenceYear,
            final String currentYear, final AffineTransform gridToWorldCorner,
            final CoordinateReferenceSystem referenceCrs, List<Geometry> rois,
            List<List<Integer>> populations, SoilSealingAdministrativeUnit sAu, 
            boolean toRasterSpace)
            throws NoSuchAuthorityCodeException, FactoryException, TransformException,
            NoninvertibleTransformException {
        boolean hasPop = true;
        if (sAu.getPopulation() != null) {
            if (sAu.getPopulation().get(referenceYear) != null) {populations.get(0).add(sAu.getPopulation().get(referenceYear));}
            else{hasPop = false;};
            if (nowFilter != null && sAu.getPopulation().get(currentYear) != null){ populations.get(1).add(sAu.getPopulation().get(currentYear));}
            else if(nowFilter != null){hasPop = false;}
        }
        if(hasPop){
        	rois.add(toReferenceCRS(sAu.getTheGeom(), referenceCrs, gridToWorldCorner, toRasterSpace));
        }
        return hasPop;
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
    protected Geometry toReferenceCRS(Geometry theGeom, CoordinateReferenceSystem referenceCrs, AffineTransform gridToWorldCorner, boolean toRasterSpace) throws NoSuchAuthorityCodeException, FactoryException, MismatchedDimensionException, TransformException, NoninvertibleTransformException {
        // check if we need to reproject the ROI from WGS84 (standard in the input) to the reference CRS
        if (theGeom.getSRID() <= 0) theGeom.setSRID(CRS.lookupEpsgCode(referenceCrs, true));
        final CoordinateReferenceSystem targetCrs = CRS.decode("EPSG:"+theGeom.getSRID(), true);
        if (CRS.equalsIgnoreMetadata(referenceCrs, targetCrs)) {
            Geometry rasterSpaceGeometry = JTS.transform(theGeom, new AffineTransform2D(gridToWorldCorner.createInverse()));
            return (toRasterSpace ? DouglasPeuckerSimplifier.simplify(rasterSpaceGeometry, 1) : theGeom);
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
            return (toRasterSpace ? JTS.transform(roiPrj, ProjectiveTransform.create(gridToWorldCorner).inverse()) : roiPrj);
        }
    }
    
    /**
     * Utility method for creating the GridGeometry for reading only the active part of the image
     * 
     * @param ciReference
     * @param rois
     * @param toRasterSpace
     * @param referenceCrs
     * @param gridROI
     * @return
     * @throws TransformException
     * @throws FactoryException
     * @throws Exception
     */
    protected GridGeometry2D createGridROI(CoverageInfo ciReference, List<Geometry> rois,
            boolean toRasterSpace, final CoordinateReferenceSystem referenceCrs,
            final Double buffer, boolean mergeGeometries)
            throws TransformException, FactoryException, Exception {
        // Creation of a Geometry union for cropping the input coverages
        Geometry union = null;
        Geometry roiUnion = null;
        Geometry roiBuffer = null;

        CoordinateReferenceSystem covCRS = referenceCrs;

        final AffineTransform gridToWorldCorner = (AffineTransform) ((GridGeometry2D) ciReference
                .getGrid()).getGridToCRS2D(PixelOrientation.UPPER_LEFT);

        // Union of all the Geometries
        for (Geometry geo : rois) {
            if (union == null) {
                if (toRasterSpace) {
                    union = JTS.transform(geo, ProjectiveTransform.create(gridToWorldCorner));
                } else {
                    union = geo;
                }
            } else {
                if (toRasterSpace) {
                    Geometry projected = JTS.transform(geo,
                            ProjectiveTransform.create(gridToWorldCorner));
                    union = union.union(projected);
                } else {
                    union = union.union(geo);
                }
            }

            if (roiUnion == null) {
                roiUnion = geo;
            } else {
                roiUnion = roiUnion.union(geo);
            }
        }

        if (mergeGeometries && !roiUnion.isEmpty()) {
            com.vividsolutions.jts.geom.Envelope envelope = roiUnion.getEnvelopeInternal();

            if (buffer != null && buffer > 0) {
                if (!"EPSG:4326".equals(ciReference.getSRS())) {
                    envelope.expandBy(buffer);
                    roiBuffer = roiUnion.buffer(buffer);
                } else {
                    envelope.expandBy(buffer / SoilSealingMiddlewareProcess.DEGREES_TO_METER_RATIO);
                    roiBuffer = roiUnion.buffer(buffer / SoilSealingMiddlewareProcess.DEGREES_TO_METER_RATIO);
                }

                if (union == null) {
                    if (toRasterSpace) {
                        union = JTS.transform(roiBuffer,
                                ProjectiveTransform.create(gridToWorldCorner));
                    } else {
                        union = roiBuffer;
                    }
                } else {
                    if (toRasterSpace) {
                        Geometry projected = JTS.transform(roiBuffer,
                                ProjectiveTransform.create(gridToWorldCorner));
                        union = union.union(projected);
                    } else {
                        union = union.union(roiBuffer);
                    }
                }
            }

            rois.clear();
            rois.add(roiUnion);
        }

        // Setting of the final srID and reproject to the final CRS
        CoordinateReferenceSystem crs = (CoordinateReferenceSystem) union.getUserData();
        if (crs != null) {
            MathTransform trans = CRS.findMathTransform(crs, covCRS);
            union = JTS.transform(union, trans);
            union.setUserData(covCRS);
        }

        if (union.getSRID() == 0) {
            int srIDfinal = CRS.lookupEpsgCode(covCRS, true);
            union.setSRID(srIDfinal);
        }

        if (union.getUserData() == null) {
            union.setUserData(covCRS);
        }

        GridGeometry2D gridROI = null;

        if (!union.isEmpty()) {

            //
            // Make sure the provided area intersects the layer BBOX in the layer CRS
            //
            final ReferencedEnvelope crsBBOX = ciReference.boundingBox();
            union = union.intersection(JTS.toGeometry(crsBBOX));
            if (union.isEmpty()) {
                throw new WPSException(
                        "The provided Administrative Areas does not intersect the reference data BBOX: ",
                        union.toText());
            }

            com.vividsolutions.jts.geom.Envelope envelope = union.getEnvelopeInternal();

            // create with supplied crs
            Envelope2D bounds = JTS.getEnvelope2D(envelope, covCRS);

            // Creation of a GridGeometry2D instance used for cropping the input images
            gridROI = new GridGeometry2D(PixelInCell.CELL_CORNER, (MathTransform) gridToWorldCorner,
                    bounds, null);
        }

        return gridROI;
    }
}