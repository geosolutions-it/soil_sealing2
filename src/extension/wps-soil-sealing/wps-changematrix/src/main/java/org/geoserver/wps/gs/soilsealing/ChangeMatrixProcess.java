/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.runtime.JCuda.cudaMemset;
import it.geosolutions.jaiext.changematrix.ChangeMatrixDescriptor;
import it.geosolutions.jaiext.changematrix.ChangeMatrixDescriptor.ChangeMatrix;
import it.geosolutions.jaiext.changematrix.ChangeMatrixRIF;

import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.awt.image.renderable.ParameterBlock;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.media.jai.JAI;
import javax.media.jai.LookupTableJAI;
import javax.media.jai.ParameterBlockJAI;
import javax.media.jai.PlanarImage;
import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;
import javax.media.jai.operator.LookupDescriptor;
import javax.media.jai.operator.TranslateDescriptor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import net.sf.json.JSONSerializer;

import org.apache.commons.io.FileUtils;
import org.geoserver.catalog.Catalog;
import org.geoserver.catalog.CoverageInfo;
import org.geoserver.config.CoverageAccessInfo;
import org.geoserver.config.GeoServer;
import org.geoserver.data.util.CoverageUtils;
import org.geoserver.wps.WPSException;
import org.geoserver.wps.area.AreaDescriptor;
import org.geoserver.wps.area.AreaRIF;
import org.geoserver.wps.gs.CoverageImporter;
import org.geoserver.wps.gs.ImportProcess;
import org.geoserver.wps.gs.ToFeature;
import org.geoserver.wps.gs.WFSLog;
import org.geoserver.wps.ppio.FeatureAttribute;
import org.geotools.coverage.Category;
import org.geotools.coverage.GridSampleDimension;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.GridCoverageFactory;
import org.geotools.coverage.grid.GridGeometry2D;
import org.geotools.coverage.grid.io.AbstractGridCoverage2DReader;
import org.geotools.coverage.grid.io.AbstractGridFormat;
import org.geotools.data.DataSourceException;
import org.geotools.data.collection.ListFeatureCollection;
import org.geotools.data.simple.SimpleFeatureCollection;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.factory.GeoTools;
import org.geotools.factory.Hints;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.gce.geotiff.GeoTiffFormat;
import org.geotools.gce.geotiff.GeoTiffWriter;
import org.geotools.gce.imagemosaic.ImageMosaicFormat;
import org.geotools.geometry.Envelope2D;
import org.geotools.geometry.jts.JTS;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.image.ImageWorker;
import org.geotools.image.jai.Registry;
import org.geotools.process.ProcessException;
import org.geotools.process.factory.DescribeParameter;
import org.geotools.process.factory.DescribeProcess;
import org.geotools.process.factory.DescribeResult;
import org.geotools.process.gs.GSProcess;
import org.geotools.referencing.CRS;
import org.geotools.referencing.crs.DefaultGeographicCRS;
import org.geotools.resources.image.ImageUtilities;
import org.geotools.util.NullProgressListener;
import org.jaitools.imageutils.ImageLayout2;
import org.opengis.coverage.grid.GridCoverage;
import org.opengis.coverage.grid.GridCoverageReader;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory;
import org.opengis.metadata.spatial.PixelOrientation;
import org.opengis.parameter.GeneralParameterValue;
import org.opengis.parameter.ParameterValueGroup;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.datum.PixelInCell;
import org.opengis.referencing.operation.MathTransform;

import com.sun.media.imageioimpl.common.ImageUtil;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.geom.PrecisionModel;

/**
 * A process that returns a coverage fully (something which is un-necessarily hard in WCS)
 * 
 * @author Simone Giannecchini, GeoSolutions SAS
 * @author Andrea Aime, GeoSolutions SAS
 */
@DescribeProcess(title = "ChangeMatrix", description = "Compute the ChangeMatrix between two coverages")
public class ChangeMatrixProcess implements GSProcess {

    static {
        Registry.registerRIF(JAI.getDefaultInstance(), new ChangeMatrixDescriptor(),
                new ChangeMatrixRIF(), Registry.JAI_TOOLS_PRODUCT);
        Registry.registerRIF(JAI.getDefaultInstance(), new AreaDescriptor(),
                new AreaRIF(), "org.soil.sealing");
    }

    // Create the PTX file by calling the NVCC
    private static String ptxfilename;

    private final static boolean DEBUG = Boolean.getBoolean("org.geoserver.wps.debug");

    private static final int PIXEL_MULTY_ARG_INDEX = 100;

    private static final int TOTAL_CLASSES = 44;

    private static final double HACONVERTER = 0.0001f;

    private static final FilterFactory ff = CommonFactoryFinder.getFilterFactory(null);

    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory(
            new PrecisionModel());

    private Catalog catalog;

    private GeoServer geoserver;

    public ChangeMatrixProcess(Catalog catalog, GeoServer geoserver) {
        this.catalog = catalog;
        this.geoserver = geoserver;
    }

    /**
     * @param classes representing the domain of the classes (Mandatory, not empty)
     * @param rasterT0 that is the reference Image (Mandatory)
     * @param rasterT1 rasterT1 that is the update situation (Mandatory)
     * @param roi that identifies the optional ROI (so that could be null)
     * @param JCUDA that indicates if the ChangeMatrix must be calculated using JCUDA or the JAI (could be null, default false)
     * @return
     */
    @DescribeResult(name = "changeMatrix", description = "the ChangeMatrix", type = ChangeMatrixDTO.class)
    public ChangeMatrixDTO execute(
            @DescribeParameter(name = "name", description = "Name of the raster, optionally fully qualified (workspace:name)") String referenceName,
            @DescribeParameter(name = "defaultStyle", description = "Name of the raster default style") String defaultStyle,
            @DescribeParameter(name = "storeName", description = "Name of the destination data store to log info") String storeName,
            @DescribeParameter(name = "typeName", description = "Name of the destination feature type to log info") String typeName,
            @DescribeParameter(name = "referenceFilter", description = "Filter to use on the raster data", min = 1) Filter referenceFilter,
            @DescribeParameter(name = "nowFilter", description = "Filter to use on the raster data", min = 1) Filter nowFilter,
            @DescribeParameter(name = "classes", collectionType = Integer.class, min = 1, description = "The domain of the classes used in input rasters") Set<Integer> classes,
            @DescribeParameter(name = "ROI", min = 0, description = "Region Of Interest") Geometry roi,
            @DescribeParameter(name = "JCUDA", min = 0, description = "Calculation of the ChangeMatrix by using JCUDA") Boolean jCudaEnabled,
            @DescribeParameter(name = "jobUid", description = "Name of the user running the Job") String jobUid)
            throws IOException {

        // DEBUG OPTION
        if (DEBUG) {
            return getTestMap();
        }

        // Check if JCUDA must be used for calculations
        boolean jcuda = false;
        if (jCudaEnabled != null) {
            jcuda = jCudaEnabled;
        }

        // get the original Coverages
        CoverageInfo ciReference = catalog.getCoverageByName(referenceName);
        if (ciReference == null) {
            throw new WPSException("Could not find coverage " + referenceName);
        }

        // ///////////////////////////////////////////////
        // ChangeMatrix outcome variables ...
        RenderedImage result = null;
        GridCoverage2D nowCoverage = null;
        GridCoverage2D referenceCoverage = null;
        ROI roiObj = null;
        // ///////////////////////////////////////////////

        // ///////////////////////////////////////////////
        // Logging to WFS variables ...
        final String wsName = ciReference.getNamespace().getPrefix();
        final UUID uuid = UUID.randomUUID();
        SimpleFeatureCollection features = null;
        Filter filter = null;
        ToFeature toFeatureProcess = new ToFeature();
        WFSLog wfsLogProcess = new WFSLog(geoserver);
        // ///////////////////////////////////////////////

        try {

            // read reference coverage
            GridCoverageReader referenceReader = ciReference.getGridCoverageReader(null, null);
            ParameterValueGroup readParametersDescriptor = referenceReader.getFormat()
                    .getReadParameters();

            // get params for this coverage and override what's needed
            Map<String, Serializable> defaultParams = ciReference.getParameters();
            GeneralParameterValue[] params = CoverageUtils.getParameters(readParametersDescriptor,
                    defaultParams, false);
            // GridGeometry object used if ROI is used
            GridGeometry2D gridROI = null;

            // handle Region Of Interest
            if (roi != null) {
                if (roi instanceof GeometryCollection) {
                    List<Polygon> geomPolys = new ArrayList<Polygon>();
                    for (int g = 0; g < ((GeometryCollection) roi).getNumGeometries(); g++) {
                        CoverageUtilities.extractPolygons(geomPolys,
                                ((GeometryCollection) roi).getGeometryN(g));
                    }

                    if (geomPolys.size() == 0) {
                        roi = GEOMETRY_FACTORY.createPolygon(null, null);
                    } else if (geomPolys.size() == 1) {
                        roi = geomPolys.get(0);
                    } else {
                        roi = roi.getFactory().createMultiPolygon(
                                geomPolys.toArray(new Polygon[geomPolys.size()]));
                    }
                }

                //
                // Make sure the provided roi intersects the layer BBOX in wgs84
                //
                final ReferencedEnvelope wgs84BBOX = ciReference.getLatLonBoundingBox();
                roi = roi.intersection(JTS.toGeometry(wgs84BBOX));
                if (roi.isEmpty()) {
                    throw new WPSException(
                            "The provided ROI does not intersect the reference data BBOX: ",
                            roi.toText());
                }

                // Geometry associated with the ROI
                Geometry roiPrj = null;

                //
                // GRID TO WORLD preparation from reference
                //
                final AffineTransform gridToWorldCorner = (AffineTransform) ((GridGeometry2D) ciReference
                        .getGrid()).getGridToCRS2D(PixelOrientation.UPPER_LEFT);

                // check if we need to reproject the ROI from WGS84 (standard in the input) to the reference CRS
                final CoordinateReferenceSystem crs = ciReference.getCRS();
                if (CRS.equalsIgnoreMetadata(crs, DefaultGeographicCRS.WGS84)) {
                    roiPrj = roi;
                    roiObj = CoverageUtilities.prepareROI2(roi, gridToWorldCorner);
                } else {
                    // reproject
                    MathTransform transform = CRS.findMathTransform(DefaultGeographicCRS.WGS84,
                            crs, true);
                    if (transform.isIdentity()) {
                        roiPrj = roi;
                    } else {
                        roiPrj = JTS.transform(roi, transform);
                    }
                    roiObj = CoverageUtilities.prepareROIGeometry(roiPrj, gridToWorldCorner);
                }
                //
                // Make sure the provided area intersects the layer BBOX in the layer CRS
                //
                final ReferencedEnvelope crsBBOX = ciReference.boundingBox();
                roiPrj = roiPrj.intersection(JTS.toGeometry(crsBBOX));
                if (roiPrj.isEmpty()) {
                    throw new WPSException(
                            "The provided ROI does not intersect the reference data BBOX: ",
                            roiPrj.toText());
                }

                // Creation of a GridGeometry object used for forcing the reader
                Envelope envelope = roiPrj.getEnvelopeInternal();
                // create with supplied crs
                Envelope2D bounds = JTS.getEnvelope2D(envelope, crs);
                // Creation of a GridGeometry2D instance used for cropping the input images
                gridROI = new GridGeometry2D(PixelInCell.CELL_CORNER,
                        (MathTransform) gridToWorldCorner, bounds, null);
            }

            // merge filter
            params = CoverageUtilities.replaceParameter(params, referenceFilter,
                    ImageMosaicFormat.FILTER);
            // merge USE_JAI_IMAGEREAD to false if needed
            params = CoverageUtilities.replaceParameter(params, !jcuda,
                    ImageMosaicFormat.USE_JAI_IMAGEREAD);
            if (gridROI != null) {
                params = CoverageUtilities.replaceParameter(params, gridROI,
                        AbstractGridFormat.READ_GRIDGEOMETRY2D);
            }
            referenceCoverage = (GridCoverage2D) referenceReader.read(params);

            if (referenceCoverage == null) {
                throw new WPSException("Input Reference Coverage not found");
            }

            // read now coverage
            readParametersDescriptor = referenceReader.getFormat().getReadParameters();
            // get params for this coverage and override what's needed
            defaultParams = ciReference.getParameters();
            params = CoverageUtils.getParameters(readParametersDescriptor, defaultParams, false);

            // merge filter
            params = CoverageUtilities
                    .replaceParameter(params, nowFilter, ImageMosaicFormat.FILTER);
            // merge USE_JAI_IMAGEREAD to false if needed
            params = CoverageUtilities.replaceParameter(params, !jcuda,
                    ImageMosaicFormat.USE_JAI_IMAGEREAD);
            if (gridROI != null) {
                params = CoverageUtilities.replaceParameter(params, gridROI,
                        AbstractGridFormat.READ_GRIDGEOMETRY2D);
            }
            // TODO add tiling, reuse standard values from config
            // TODO add background value, reuse standard values from config
            nowCoverage = (GridCoverage2D) referenceReader.read(params);

            if (nowCoverage == null) {
                throw new WPSException("Input Current Coverage not found");
            }

            // Definition of the final raster name
            final String rasterName = ciReference.getName() + "_cm_" + System.nanoTime();

            // //////////////////////////////////////////////////////////////////////
            // Logging to WFS ...
            // //////////////////////////////////////////////////////////////////////
            /**
             * Convert the spread attributes into a FeatureType
             */
            List<FeatureAttribute> attributes = new ArrayList<FeatureAttribute>();

            attributes.add(new FeatureAttribute("ftUUID", uuid.toString()));
            attributes.add(new FeatureAttribute("runBegin", new Date()));
            attributes.add(new FeatureAttribute("runEnd", new Date()));
            attributes.add(new FeatureAttribute("itemStatus", "RUNNING"));
            attributes.add(new FeatureAttribute("itemStatusMessage", "Instrumented by Server"));
            attributes.add(new FeatureAttribute("referenceName", referenceName));
            attributes.add(new FeatureAttribute("defaultStyle", defaultStyle));
            attributes.add(new FeatureAttribute("referenceFilter", referenceFilter.toString()));
            attributes.add(new FeatureAttribute("nowFilter", nowFilter.toString()));
            attributes.add(new FeatureAttribute("roi", roi.getEnvelope()));
            attributes.add(new FeatureAttribute("wsName", wsName));
            attributes.add(new FeatureAttribute("layerName", ""));
            attributes.add(new FeatureAttribute("changeMatrix", ""));
            attributes.add(new FeatureAttribute("jcuda", (jcuda ? "[CUDA]" : "[JAVA]")));
            attributes.add(new FeatureAttribute("jobUid", jobUid));

            features = toFeatureProcess.execute(JTS.toGeometry(ciReference.getNativeBoundingBox()),
                    ciReference.getCRS(), typeName, attributes, null);

            if (features == null || features.isEmpty()) {
                throw new ProcessException("There was an error while converting attributes into FeatureType.");
            }

            /**
             * LOG into the DB
             */
            filter = ff.equals(ff.property("ftUUID"), ff.literal(uuid.toString()));
            features = wfsLogProcess.execute(features, typeName, wsName, storeName, filter, true, new NullProgressListener());

            if (features == null || features.isEmpty()) {
                throw new ProcessException("There was an error while logging FeatureType into the storage.");
            }

            // //////////////////////////////////////////////////////////////////////
            // Compute the Change Matrix ...
            // //////////////////////////////////////////////////////////////////////

            String refYear = null;
            String nowYear = null;

            Pattern pattern = Pattern.compile("(\\d{4}?)");
            if (referenceFilter != null) {
                Matcher matcher = pattern.matcher(referenceFilter.toString());
                if (matcher.find()) {
                    refYear = matcher.group(1);
                }
            }

            if (nowFilter != null) {
                Matcher matcher = pattern.matcher(nowFilter.toString());
                if (matcher.find()) {
                    nowYear = matcher.group(1);
                }
            }
            // Calculation of the Area Image
            GridGeometry2D gg2D = referenceCoverage.getGridGeometry();
            ParameterBlock pb = new ParameterBlock();
            pb.setSource(referenceCoverage.getRenderedImage(), 0);
            pb.set(new ReferencedEnvelope(gg2D.getEnvelope()), 0);
            pb.set(HACONVERTER, 1);
            pb.set(classes, 2);
            if(roiObj != null){
                pb.set(roiObj, 3);
            }
            RenderedOp areaImage = JAI.create("area", pb);

            // Selection of the Object used for calculating the ChangeMatrix
            ChangeMatrixCalculator calculator = ChangeMatrixCalculator.getCalculator(jcuda);
            // Calculation of the ChangeMatrix
            ChangeMatrixContainer container = calculator.computeChangeMatrix(geoserver,
                    referenceCoverage.getRenderedImage(), nowCoverage.getRenderedImage(),
                    areaImage, classes, roiObj, rasterName, refYear, nowYear);

            // Setting of the results
            result = container.getResult();
            ChangeMatrixDTO changeMatrix = container.getDto();

            // //////////////////////////////////////////////////////////////////////
            // Import into GeoServer the new raster 'result' ...
            // //////////////////////////////////////////////////////////////////////
            /**
             * create the final coverage using final envelope
             */
            // hints for tiling
            final Hints hints = GeoTools.getDefaultHints().clone();
            
            // build the output sample dimensions, use the default value ( 0 ) as the no data
            final GridSampleDimension outSampleDimension = new GridSampleDimension("classification",
                    new Category[] { Category.NODATA }, null).geophysics(true);
            
            final GridCoverage2D retValue = new GridCoverageFactory(hints).create(rasterName,
                    result, referenceCoverage.getEnvelope(), 
                    new GridSampleDimension[] { outSampleDimension },
                    new GridCoverage[] { referenceCoverage }, new HashMap<String,Double>(){{
                        put("GC_NODATA", 0d);
                      }});
            
            /**
             * Add Overviews...
             */
            final File file = File.createTempFile(retValue.getName().toString(), ".tif");
            GeoTiffWriter writer = new GeoTiffWriter(file);

            // setting the write parameters for this geotiff
            final ParameterValueGroup gtiffParams = new GeoTiffFormat().getWriteParameters();
            gtiffParams.parameter(AbstractGridFormat.GEOTOOLS_WRITE_PARAMS.getName().toString())
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

            // Disposal of the input Raster
            PlanarImage.wrapRenderedImage(result).dispose();

            AbstractGridCoverage2DReader gtiffReader = null;
            try {
                gtiffReader = new GeoTiffFormat().getReader(file);
                CoverageUtilities.generateOverviews(gtiffReader);
            } catch (DataSourceException e) {
                // we tried, no need to fuss around this one
            }

            /**
             * import the new coverage into the GeoServer catalog
             */
            try {
                ImportProcess importProcess = new ImportProcess(catalog);
                GridCoverage2D retOvValue = gtiffReader.read(wps);
                 importProcess.execute(null, retOvValue, wsName, null, retValue.getName().toString(), retValue.getCoordinateReferenceSystem(), null,
                 defaultStyle);
            } finally {
                if (gtiffReader != null) {
                    gtiffReader.dispose();
                }

                try {
                     FileUtils.forceDelete(file);
                } catch (Exception e) {
                    // we tried, no need to fuss around this one
                }
            }

            // //////////////////////////////////////////////////////////////////////
            // Updating WFS ...
            // //////////////////////////////////////////////////////////////////////
            /**
             * Update Feature Attributes and LOG into the DB
             */
            filter = ff.equals(ff.property("ftUUID"), ff.literal(uuid.toString()));

            SimpleFeature feature = SimpleFeatureBuilder.copy(features.subCollection(filter)
                    .toArray(new SimpleFeature[1])[0]);

            // build the feature
            feature.setAttribute("runEnd", new Date());
            feature.setAttribute("itemStatus", "COMPLETED");
            feature.setAttribute("itemStatusMessage",
                    "Change Matrix Process completed successfully");
            feature.setAttribute("layerName", rasterName);
            feature.setAttribute("changeMatrix", JSONSerializer.toJSON(changeMatrix).toString());

            ListFeatureCollection output = new ListFeatureCollection(features.getSchema());
            output.add(feature);

            features = wfsLogProcess.execute(output, typeName, wsName, storeName, filter, false,
                    new NullProgressListener());

            // //////////////////////////////////////////////////////////////////////
            // Return the computed Change Matrix ...
            // //////////////////////////////////////////////////////////////////////
            return changeMatrix;
        } catch (Exception e) {

            if (features != null) {
                // //////////////////////////////////////////////////////////////////////
                // Updating WFS ...
                // //////////////////////////////////////////////////////////////////////
                /**
                 * Update Feature Attributes and LOG into the DB
                 */
                filter = ff.equals(ff.property("ftUUID"), ff.literal(uuid.toString()));

                SimpleFeature feature = SimpleFeatureBuilder.copy(features.subCollection(filter)
                        .toArray(new SimpleFeature[1])[0]);

                // build the feature
                feature.setAttribute("runEnd", new Date());
                feature.setAttribute("itemStatus", "FAILED");
                feature.setAttribute(
                        "itemStatusMessage",
                        "There was an error while while processing Input parameters: "
                                + e.getMessage());

                ListFeatureCollection output = new ListFeatureCollection(features.getSchema());
                output.add(feature);

                features = wfsLogProcess.execute(output, typeName, wsName, storeName, filter,
                false, new NullProgressListener());
            }

            throw new WPSException("Could process request ", e);
        } finally {
            // clean up
            if (result != null) {
                ImageUtilities.disposePlanarImageChain(PlanarImage.wrapRenderedImage(result));
            }
            if (referenceCoverage != null) {
                referenceCoverage.dispose(true);
            }
            if (nowCoverage != null) {
                nowCoverage.dispose(true);
            }
        }
    }

    /**
     * @return an hardcoded ChangeMatrixOutput usefull for testing
     */
    private static final ChangeMatrixDTO getTestMap() {

        ChangeMatrixDTO s = new ChangeMatrixDTO();

        s.add(new ChangeMatrixElement(0, 0, 16002481));
        s.add(new ChangeMatrixElement(0, 35, 0));
        s.add(new ChangeMatrixElement(0, 1, 0));
        s.add(new ChangeMatrixElement(0, 36, 4));
        s.add(new ChangeMatrixElement(0, 37, 4));

        s.add(new ChangeMatrixElement(1, 0, 0));
        s.add(new ChangeMatrixElement(1, 35, 0));
        s.add(new ChangeMatrixElement(1, 1, 3192));
        s.add(new ChangeMatrixElement(1, 36, 15));
        s.add(new ChangeMatrixElement(1, 37, 0));

        s.add(new ChangeMatrixElement(35, 0, 0));
        s.add(new ChangeMatrixElement(35, 35, 7546));
        s.add(new ChangeMatrixElement(35, 1, 0));
        s.add(new ChangeMatrixElement(35, 36, 0));
        s.add(new ChangeMatrixElement(35, 37, 16));

        s.add(new ChangeMatrixElement(36, 0, 166));
        s.add(new ChangeMatrixElement(36, 35, 36));
        s.add(new ChangeMatrixElement(36, 1, 117));
        s.add(new ChangeMatrixElement(36, 36, 1273887));
        s.add(new ChangeMatrixElement(36, 37, 11976));

        s.add(new ChangeMatrixElement(37, 0, 274));
        s.add(new ChangeMatrixElement(37, 35, 16));
        s.add(new ChangeMatrixElement(37, 1, 16));
        s.add(new ChangeMatrixElement(37, 36, 28710));
        s.add(new ChangeMatrixElement(37, 37, 346154));

        return s;
    }

    public static String getPtxfilename() {
        return ptxfilename;
    }

    public static void setPtxfilename(String ptxfilename) {
        ChangeMatrixProcess.ptxfilename = ptxfilename;
    }

    /**
     * Container class used for packing the output of the {@link ChangeMatrixCalculator} class in a single object.
     */
    static class ChangeMatrixContainer {
        public RenderedImage getResult() {
            return result;
        }

        public void setResult(RenderedImage result) {
            this.result = result;
        }

        public ChangeMatrixDTO getDto() {
            return dto;
        }

        public void setDto(ChangeMatrixDTO dto) {
            this.dto = dto;
        }

        /** RenderedImage created by the {@link ChangeMatrixCalculator} class */
        private RenderedImage result;

        /** {@link ChangeMatrixDTO} object containing the ChangeMatrix */
        private ChangeMatrixDTO dto;
    }

    /**
     * Enum used for calculating the ChangeMatrix
     */
    public enum ChangeMatrixCalculator {
        JAIEXT {
            @Override
            public ChangeMatrixContainer computeChangeMatrix(GeoServer geoserver,
                    RenderedImage ref, RenderedImage cur, RenderedImage area,
                    Set<Integer> usedClass, ROI roi, String rasterName, String refYear,
                    String nowYear) {
                // ParameterBlock object
                final ParameterBlockJAI pbj = new ParameterBlockJAI("ChangeMatrix");
                // ChangeMatrix Object
                final ChangeMatrix cm = new ChangeMatrix(usedClass);
                pbj.setParameter("result", cm);
                pbj.setParameter(
                        ChangeMatrixDescriptor.PARAM_NAMES[ChangeMatrixDescriptor.PIXEL_MULTY_ARG_INDEX],
                        PIXEL_MULTY_ARG_INDEX);
                pbj.setParameter(
                        ChangeMatrixDescriptor.PARAM_NAMES[ChangeMatrixDescriptor.AREA_MAP_INDEX],
                        area);
                // Setting of the ROI parameter for the JAI
                if (roi != null) {
                    pbj.setParameter("ROI", roi);
                }
                // Setting of the sources
                pbj.addSource(ref);
                pbj.addSource(cur);
                // ChangeMatrix creation
                RenderedOp result = JAI.create("ChangeMatrix", pbj, null);

                //
                // result computation
                //
                final int numTileX = result.getNumXTiles();
                final int numTileY = result.getNumYTiles();
                final int minTileX = result.getMinTileX();
                final int minTileY = result.getMinTileY();
                final List<Point> tiles = new ArrayList<Point>(numTileX * numTileY);
                for (int i = minTileX; i < minTileX + numTileX; i++) {
                    for (int j = minTileY; j < minTileY + numTileY; j++) {
                        tiles.add(new Point(i, j));
                    }
                }
                final CountDownLatch sem = new CountDownLatch(tiles.size());
                // how many JAI tiles do we have?
                final CoverageAccessInfo coverageAccess = geoserver.getGlobal().getCoverageAccess();
                final ThreadPoolExecutor executor = coverageAccess.getThreadPoolExecutor();
                final RenderedOp temp = result;
                for (final Point tile : tiles) {

                    executor.execute(new Runnable() {

                        @Override
                        public void run() {
                            temp.getTile(tile.x, tile.y);
                            sem.countDown();
                        }
                    });
                }
                try {
                    sem.await();
                } catch (InterruptedException e) {
                    // TODO handle error
                    return null;
                }
                // computation done!
                cm.freeze();

                // ImageDisposal
                if(area instanceof RenderedOp){
                    ((RenderedOp)area).dispose();
                }

                // Creation of the DTO
                final ChangeMatrixDTO changeMatrix = new ChangeMatrixDTO(cm, usedClass, rasterName,
                        refYear, nowYear);

                // Creation of the final Container
                ChangeMatrixContainer container = new ChangeMatrixContainer();
                container.setDto(changeMatrix);
                container.setResult(result);
                return container;
            }
        },
        JCUDA {
            @Override
            public ChangeMatrixContainer computeChangeMatrix(GeoServer geoserver,
                    RenderedImage ref, RenderedImage cur, RenderedImage area,
                    Set<Integer> usedClass, ROI roi, String rasterName, String refYear,
                    String nowYear) {

                // Selection of the Bounds of the input data
                Rectangle rectIMG = new Rectangle(ref.getMinX(), ref.getMinY(), ref.getWidth(),
                        ref.getHeight());

                ImageWorker wRef = new ImageWorker(ref);
                ImageWorker wCur = new ImageWorker(cur);

                // Selection of the input images as BufferedImage
                BufferedImage reference = wRef.getBufferedImage();
                BufferedImage current = wCur.getBufferedImage();
                // ImageWorker disposal
                wRef.dispose();
                wCur.dispose();

                // Copy the input SET
                Set<Integer> classes = new TreeSet<Integer>(usedClass);

                // Check if the 0 class is not present. If so the 0 class is added
                boolean zeroAdd = classes.add(0);

                // transform into byte array
                Raster data = reference.getData();
                final DataBufferByte dbRef = (DataBufferByte) data.getDataBuffer();
                Raster data2 = current.getData();
                final DataBufferByte dbCurrent = (DataBufferByte) data2.getDataBuffer();
                byte dataRef[] = dbRef.getData();
                byte dataCurrent[] = dbCurrent.getData();

                // Flush of the input BufferedImages
                reference.flush();
                current.flush();

                // Create ROI with the same dimensions of the images
                byte[] dataROI = null;
                if (roi != null) {
                    dataROI = getROIData(roi, rectIMG);
                } else {
                    // Create an array of 1
                    dataROI = new byte[dataRef.length];
                    Arrays.fill(dataROI, (byte) 1);
                }

                // Computation of the ChangeMatrix using JCUDA
                final List<int[]> resultCuda = JCudaChangeMat(dataRef, dataCurrent, dataROI,
                        TOTAL_CLASSES, rectIMG.width, rectIMG.height);

                // Image creation from data
                BufferedImage output = createImage(rectIMG, resultCuda.get(0));
                RenderedImage result = output;
                // Translation of the BufferedImage from (0,0) to (rectIMG.x, rectIMG.y)
                if (rectIMG.x != 0 || rectIMG.y != 0) {
                    result = TranslateDescriptor.create(output, rectIMG.x * 1.0f, rectIMG.y * 1.0f,
                            null, GeoTools.getDefaultHints());
                }

                /**
                 * creating the ChangeMatrix grid
                 */
                final ChangeMatrixDTO changeMatrix = new ChangeMatrixDTO();

                int[] changeMat = resultCuda.get(1);

                if (zeroAdd) {
                    classes.remove(0);
                }

                for (int i = 0; i < TOTAL_CLASSES; i++) {
                    for (int j = 0; j < TOTAL_CLASSES; j++) {
                        // Only the class taken into account are selected
                        if (classes.contains(i) && classes.contains(j)) {
                            // Value
                            int index = i + j * TOTAL_CLASSES;
                            int classValue = changeMat[index];
                            //TODO AREA MUST BE CALCULATED FOR JCUDA
                            ChangeMatrixElement el = new ChangeMatrixElement(i, j, classValue);
                            changeMatrix.add(el);
                        }
                    }
                }
                // Raster name and Year setting
                changeMatrix.setRasterName(rasterName);
                changeMatrix.setRefYear(refYear);
                changeMatrix.setNowYear(nowYear);

                // Creation of the container
                ChangeMatrixContainer container = new ChangeMatrixContainer();
                container.setDto(changeMatrix);
                container.setResult(result);
                return container;
            }

            private byte[] getROIData(ROI roi, Rectangle rectIMG) {
                byte[] dataROI;
                PlanarImage roiIMG = roi.getAsImage();
                Rectangle rectROI = roiIMG.getBounds();
                // Forcing to component colormodel in order to avoid packed bits
                ImageWorker w = new ImageWorker();
                w.setImage(roiIMG);
                w.forceComponentColorModel();
                RenderedImage img = w.getRenderedImage();
                //
                BufferedImage test = new BufferedImage(rectIMG.width, rectIMG.height,
                        BufferedImage.TYPE_BYTE_GRAY);
                ImageLayout2 layout = new ImageLayout2(test);
                layout.setMinX(img.getMinX());
                layout.setMinY(img.getMinY());
                layout.setWidth(img.getWidth());
                layout.setHeight(img.getHeight());
                // Lookup
                byte[] lut = new byte[256];
                lut[255] = 1;
                lut[1] = 1;
                LookupTableJAI table = new LookupTableJAI(lut);
                RenderingHints hints = new RenderingHints(JAI.KEY_IMAGE_LAYOUT, layout);
                RenderedOp transformed = LookupDescriptor.create(img, table, hints);

                Graphics2D gc2d = null;
                // Translation parameters in order to position the ROI data correctly in the Raster Space
                int trX = -rectIMG.x + rectROI.x - rectIMG.x;
                int trY = -rectIMG.y + rectROI.y - rectIMG.y;
                try {
                    gc2d = test.createGraphics();
                    gc2d.drawRenderedImage(transformed,
                            AffineTransform.getTranslateInstance(trX, trY));
                } finally {
                    gc2d.dispose();
                }
                Rectangle testRect = new Rectangle(rectIMG.width, rectIMG.height);
                DataBufferByte dbRoi = (DataBufferByte) test.getData(testRect).getDataBuffer();
                dataROI = dbRoi.getData();
                // BufferedImage is stored in memory so the planarImage chain before can be disposed
                ImageUtilities.disposePlanarImageChain(transformed);
                // Flush of the BufferedImage
                test.flush();

                return dataROI;
            }

            /**
             * Stub method to be replaced with CUDA code
             * 
             * @param host_iMap1 reference map, nodata=0 [uint8]
             * @param host_iMap2 current map, nodata=0 [uint8]
             * @param host_roiMap roi map, 1->to-be-counted & 0->ignoring pixels [uint8]
             * @param numclasses number of classes including zero (which is NODATA value)
             * @param imWidth number of pixels of *iMap* along X
             * @param imHeight number of pixels of *iMap* along Y
             * @return a list of uint32 arrays containing (1) oMap and (2) changeMatrix
             */
            private List<int[]> JCudaChangeMat(byte[] host_iMap1, byte[] host_iMap2,
                    byte[] host_roiMap, int numclasses, int imWidth, int imHeight) {
                /*
                 * Copyright 2013 Massimo Nicolazzo & Giuliano Langella: ---- completed ----
                 * 	(1) kernel-1 --->{oMap, tiled changeMat}
                 * 	(2) kernel-2 --->{changeMat}
                 */

                /**
                 * This uses the JCuda driver bindings to load and execute two CUDA kernels:
                 * (1) The first kernel executes the change matrix computation for the whole
                 * ROI given as input in the form of iMap1 & iMap2. It returns a 3D change
                 * matrix in which every 2D array corresponds to a given CUDA-tile (not GIS-tile).
                 * (2) The second kernel sum up the 3D change matrix returning one 2D array
                 * being the accountancy for the whole ROI.
                 */

                // Enable exceptions and omit all subsequent error checks
                JCudaDriver.setExceptionsEnabled(true);

                // Definition of the array dimensions in bytes
                int imap_bytes = imWidth * imHeight * Sizeof.BYTE; // uint8_t
                int omap_bytes = imWidth * imHeight * Sizeof.INT; // uint32_t
                int chmat_bytes = numclasses * numclasses * imHeight * Sizeof.INT; // uint32_t

                // Initialise the driver:
                cuInit(0);

                // Obtain the number of devices:
                int gpuDeviceCount[] = { 0 };
                cuDeviceGetCount(gpuDeviceCount);
                int deviceCount = gpuDeviceCount[0];
                if (deviceCount == 0) {
                    throw new ProcessException("error: no devices supporting CUDA.");
                }

                // Select first device, but I should select/split devices
                int selDev = 0;
                CUdevice device = new CUdevice();
                cuDeviceGet(device, selDev);

                // Get some useful properties:
                int amountProperty[] = { 0 };
                // -1-
                cuDeviceGetAttribute(amountProperty, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                        device);
                int maxThreadsPerBlock = amountProperty[0];

                // Create a context for the selected device
                CUcontext context = new CUcontext();
                // int cuCtxCreate_STATUS =
                cuCtxCreate(context, selDev, device);

                // Load the ptx file.
                CUmodule module = new CUmodule();
                cuModuleLoad(module, ptxfilename);

                // Obtain a function pointer to the "add" function.
                CUfunction changemap = new CUfunction();
                cuModuleGetFunction(changemap, module, "_Z9changemapPKhS0_S0_jjjjPjS1_");
                CUfunction changemat = new CUfunction();
                cuModuleGetFunction(changemat, module, "_Z9changematPjjj");

                // Allocate the device input data, and copy the
                // host input data to the device
                CUdeviceptr dev_iMap1 = new CUdeviceptr();
                cuMemAlloc(dev_iMap1, imap_bytes);
                cuMemcpyHtoD(dev_iMap1, Pointer.to(host_iMap1), imap_bytes);

                CUdeviceptr dev_iMap2 = new CUdeviceptr();
                cuMemAlloc(dev_iMap2, imap_bytes);
                cuMemcpyHtoD(dev_iMap2, Pointer.to(host_iMap2), imap_bytes);

                CUdeviceptr dev_roiMap = null;
                if (host_roiMap != null) {
                    dev_roiMap = new CUdeviceptr();
                    cuMemAlloc(dev_roiMap, imap_bytes);
                    cuMemcpyHtoD(dev_roiMap, Pointer.to(host_roiMap), imap_bytes);
                }

                // Allocate device output memory
                CUdeviceptr dev_oMap = new CUdeviceptr();
                cuMemAlloc(dev_oMap, omap_bytes);

                CUdeviceptr dev_chMat = new CUdeviceptr();
                cuMemAlloc(dev_chMat, chmat_bytes);// ERROR with Integer.SIZE

                cudaMemset(dev_chMat, 0, chmat_bytes);

                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParameters1 = Pointer.to(Pointer.to(dev_iMap1),
                        Pointer.to(dev_iMap2), Pointer.to(dev_roiMap),
                        Pointer.to(new int[] { imWidth }), Pointer.to(new int[] { imHeight }),
                        Pointer.to(new int[] { imWidth }), Pointer.to(new int[] { numclasses }),
                        Pointer.to(dev_chMat), Pointer.to(dev_oMap));

                // Call the kernel function.
                int blockSizeX = (int) Math.floor(Math.sqrt(maxThreadsPerBlock)); // 32
                int blockSizeY = (int) Math.floor(Math.sqrt(maxThreadsPerBlock)); // 32
                int blockSizeZ = 1;
                int gridSizeX = 1 + (int) Math.ceil(imHeight / (blockSizeX * blockSizeY));
                int gridSizeY = 1;
                int gridSizeZ = 1;
                // Launch Kernel 1
                cuLaunchKernel(changemap, gridSizeX, gridSizeY, gridSizeZ, // Grid dimension
                        blockSizeX, blockSizeY, blockSizeZ, // Block dimension
                        0, null, // Shared memory size and stream
                        kernelParameters1, null // Kernel- and extra parameters
                );
                // Wait the end of the computations
                cuCtxSynchronize();

                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParameters2 = Pointer.to(Pointer.to(dev_chMat),
                        Pointer.to(new int[] { numclasses * numclasses }),
                        Pointer.to(new int[] { imHeight }));
                // Launch Kernel 2
                cuLaunchKernel(changemat, gridSizeX, gridSizeY, gridSizeZ, // Grid dimension
                        blockSizeX, blockSizeY, blockSizeZ, // Block dimension
                        0, null, // Shared memory size and stream
                        kernelParameters2, null // Kernel- and extra parameters
                );
                // Wait the end of the computations
                cuCtxSynchronize();

                // Allocate host output memory and copy the device output
                // to the host.
                int host_chMat[] = new int[numclasses * numclasses];
                cuMemcpyDtoH(Pointer.to(host_chMat), dev_chMat, chmat_bytes / imHeight);
                int host_oMap[] = new int[imHeight * imWidth];
                cuMemcpyDtoH(Pointer.to(host_oMap), dev_oMap, omap_bytes);

                // Clean up.
                cuMemFree(dev_iMap1);
                cuMemFree(dev_iMap2);
                cuMemFree(dev_roiMap);
                cuMemFree(dev_oMap);
                cuMemFree(dev_chMat);

                // Unload MODULE
                cuModuleUnload(module);

                // Destroy CUDA context:
                cuCtxDestroy(context);

                // OUTPUT:
                return Arrays.asList(host_oMap, host_chMat);
            }

            /**
             * Creates an image from an array of {@link Integer}
             * 
             * @param rect
             * @param data
             * @return
             */
            private BufferedImage createImage(Rectangle rect, final int[] data) {
                // Definition of the SampleModel
                final SampleModel sm = new PixelInterleavedSampleModel(DataBuffer.TYPE_INT,
                        rect.width, rect.height, 1, rect.width, new int[] { 0 });
                // DataBuffer containing input data
                final DataBufferInt db1 = new DataBufferInt(data, rect.width * rect.height);
                // Writable Raster used for creating the BufferedImage
                final WritableRaster wr = com.sun.media.jai.codecimpl.util.RasterFactory
                        .createWritableRaster(sm, db1, new Point(0, 0));
                final BufferedImage image = new BufferedImage(ImageUtil.createColorModel(sm), wr,
                        false, null);
                return image;
            }
        };

        /**
         * Computes the ChangeMatrix using the provided parameters. This method returns a {@link ChangeMatrixContainer} object which stores internally
         * the output image and the changematrix
         * 
         * @param geoserver
         * @param ref
         * @param cur
         * @param area
         * @param usedClass
         * @param roi
         * @param rasterName
         * @param nowYear
         * @param refYear
         * @return
         */
        public abstract ChangeMatrixContainer computeChangeMatrix(GeoServer geoserver,
                RenderedImage ref, RenderedImage cur, RenderedImage area, Set<Integer> usedClass,
                ROI roi, String rasterName, String refYear, String nowYear);

        /**
         * Returns a calculator for JCUDA or the JAI operation
         * 
         * @param jcuda
         * @return
         */
        public static ChangeMatrixCalculator getCalculator(boolean jcuda) {
            if (jcuda) {
                return JCUDA;
            } else {
                return JAIEXT;
            }
        }
    }
}
