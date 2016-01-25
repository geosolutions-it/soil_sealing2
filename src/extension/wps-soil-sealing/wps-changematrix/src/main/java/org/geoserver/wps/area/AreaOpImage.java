package org.geoserver.wps.area;

import java.awt.Rectangle;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.media.jai.ImageLayout;
import javax.media.jai.PointOpImage;
import javax.media.jai.ROI;
import javax.media.jai.ROIShape;
import javax.media.jai.iterator.RectIter;
import javax.media.jai.iterator.RectIterFactory;

import org.geotools.coverage.grid.GridEnvelope2D;
import org.geotools.coverage.grid.GridGeometry2D;
import org.geotools.geometry.jts.JTS;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.referencing.CRS;
import org.jaitools.imageutils.ImageLayout2;
import org.opengis.coverage.grid.GridEnvelope;
import org.opengis.metadata.spatial.PixelOrientation;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.MathTransform2D;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LinearRing;
import com.vividsolutions.jts.geom.Polygon;

/**
 * This class calculates the area for each pixel of the input image, providing an associated envelope. An optional ROI can be used for reducing the
 * computation. Also the user can define a set of valid values of the input image on which calculating the area. Note that a pixel with value 0 is
 * skipped. The operation can be executed only on integer images in order to be able to handle the valid values.
 */
public class AreaOpImage extends PointOpImage {

    /** Eckert IV wkt for reprojecting into an Equal area region */
    private static final String TARGET_CRS_WKT = "PROJCS[\"World_Eckert_IV\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Eckert_IV\"],PARAMETER[\"Central_Meridian\",0.0],UNIT[\"Meter\",1.0]]";

    /** Logger to use for logging the exceptions */
    private static final Logger LOGGER = Logger.getLogger(AreaOpImage.class.toString());

    /** Multiplier value to multiply the area value */
    private final double multi;

    /** Set of valid values */
    private final Set<Integer> validValues;

    /** Flag indicating that validvalues must be checked */
    private final boolean validCheck;

    /** Input envelope */
    private final ReferencedEnvelope envelope;

    /** Transformation from Raster to Model space */
    private final MathTransform2D g2w;

    /** Transformation from Input envelope crs to Eckert IV */
    private final MathTransform transform;

    /** Optional ROI used for reducing the computation area */
    private ROI roiUsed;

    /** Flag indicating the presence of ROI */
    private final boolean noROI;

    /** Geometry factory used */
    private static final GeometryFactory GEOM_FACTORY = new GeometryFactory();

    public AreaOpImage(RenderedImage source, ImageLayout layout, Map configuration,
            ReferencedEnvelope env, double multiplier, Set<Integer> validValues, ROI roi) {
        super(source, layoutHelper(source, layout), configuration, true);

        this.multi = multiplier;
        this.validCheck = validValues != null && !validValues.isEmpty();
        this.validValues = validValues;
        this.envelope = env;

        // Creation of a GridGeometry in order to calculate the gridToWorld transform
        GridEnvelope gridRange = new GridEnvelope2D(getBounds());
        GridGeometry2D gg = new GridGeometry2D(gridRange, env);
        g2w = gg.getGridToCRS2D(PixelOrientation.UPPER_LEFT);

        CoordinateReferenceSystem sourceCRS = envelope.getCoordinateReferenceSystem();

        try {
            CoordinateReferenceSystem targetCRS = CRS.parseWKT(TARGET_CRS_WKT);
            transform = CRS.findMathTransform(sourceCRS, targetCRS);
        } catch (FactoryException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IllegalArgumentException(e);
        }

        // Setting of the roi
        this.roiUsed = roi;
        if (roi != null) {
            // Setting a roi flag to true
            this.noROI = false;
            // check that the ROI contains the source image bounds
            final Rectangle sourceBounds = new Rectangle(source.getMinX(), source.getMinY(),
                    source.getWidth(), source.getHeight());
            // Check if the ROI intersects the image bounds
            if (!roi.intersects(sourceBounds)) {
                throw new IllegalArgumentException(
                        "The bounds of the ROI must intersect the source image");
            }
            // massage roi
            roiUsed = roi.intersect(new ROIShape(sourceBounds));
        } else {
            this.noROI = true;
        }
    }

    /**
     * Preparation of the image layout by setting the sample model data type to double.
     * 
     * @param source
     * @param layout
     * @return
     */
    private static ImageLayout layoutHelper(RenderedImage source, ImageLayout layout) {

        ImageLayout il = null;
        // Check if it is already present
        if (layout != null) {
            il = (ImageLayout) layout.clone();
        } else {
            il = new ImageLayout2(source);
        }
        // Setting of the new SampleModel
        SampleModel sampleModel = new ComponentSampleModel(DataBuffer.TYPE_DOUBLE,
                source.getWidth(), source.getHeight(), 1, source.getWidth(), new int[] { 0 });

        if (il.isValid(ImageLayout.SAMPLE_MODEL_MASK)) {
            il.unsetValid(ImageLayout.SAMPLE_MODEL_MASK);
        }
        il.setSampleModel(sampleModel);

        return il;

    }

    @Override
    protected void computeRect(Raster[] sources, WritableRaster dest, Rectangle destRect) {

        // massage roi
        ROI tileRoi = null;
        if (!noROI) {
            Rectangle roiRect = destRect.getBounds();
            // Expand tile dimensions
            roiRect.grow(1, 1);
            tileRoi = roiUsed.intersect(new ROIShape(roiRect));
        }

        if (noROI || !tileRoi.getBounds().isEmpty()) {
            // Source tile
            Raster source = sources[0];
            // Envelope of the tile in the Raster space
            ReferencedEnvelope tileEnv = new ReferencedEnvelope(destRect.getMinX(),
                    destRect.getMaxX(), destRect.getMinY(), destRect.getMaxY(), null);
            Envelope env = null;
            try {
                // Transform to the model space
                env = JTS.transform(tileEnv, g2w);
            } catch (TransformException e) {
                LOGGER.log(Level.SEVERE, e.getMessage(), e);
                return;
            }

            // Iterator on the input tile
            RectIter iter = RectIterFactory.create(source, destRect);

            int originX = destRect.x;
            int originY = destRect.y;
            int width = destRect.width;
            int height = destRect.height;

            try {
                Polygon polygon = null;
                // Calculation of the iteration steps
                double pX = env.getMinX();
                double pY = env.getMaxY();
                double stepX = (env.getMaxX() - env.getMinX()) / width;
                double stepY = (env.getMaxY() - env.getMinY()) / height;

                Coordinate[] tempCoordinates = new Coordinate[5];

                iter.startBands();
                iter.startLines();
                // Cycle on all the input tile
                int i = 0;
                while (!iter.finishedLines()) {
                    iter.startPixels();
                    int y = originY + i;
                    // start of the row
                    pX = env.getMinX();
                    int j = 0;
                    while (!iter.finishedPixels()) {
                        int x = originX + j;
                        int sample = iter.getSample(0);
                        // Check if the pixel value is valid and if it is inside the ROI
                        boolean validValue = sample != 0
                                && ((validCheck && validValues.contains(sample)) || !validCheck);
                        boolean inROI = noROI || tileRoi.contains(x, y);
                        if (validValue && inROI) {
                            double nX = pX + stepX;
                            double nY = pY - stepY;

                            // Creation of a new Polygon or update of the previous one
                            if (polygon == null) {
                                tempCoordinates[0] = new Coordinate(pX, pY);
                                tempCoordinates[1] = new Coordinate(nX, pY);
                                tempCoordinates[2] = new Coordinate(nX, nY);
                                tempCoordinates[3] = new Coordinate(pX, nY);
                                tempCoordinates[4] = tempCoordinates[0];
                                LinearRing linearRing = GEOM_FACTORY
                                        .createLinearRing(tempCoordinates);
                                polygon = GEOM_FACTORY.createPolygon(linearRing, null);
                            } else {
                                tempCoordinates[0].x = pX;
                                tempCoordinates[0].y = pY;
                                tempCoordinates[1].x = nX;
                                tempCoordinates[1].y = pY;
                                tempCoordinates[2].x = nX;
                                tempCoordinates[2].y = nY;
                                tempCoordinates[3].x = pX;
                                tempCoordinates[3].y = nY;
                                polygon.geometryChanged();
                            }

                            // transform to EckertIV and compute area
                            Geometry targetGeometry = JTS.transform(polygon, transform);
                            // Set the values to the image
                            double area = targetGeometry.getArea() * multi;
                            if (area > 0) {
                                dest.setSample(x, y, 0, area);
                            }
                        }

                        // move on
                        pX = pX + stepX;
                        j++;
                        iter.nextPixelDone();
                    }
                    // move to next row
                    pY = pY - stepY;
                    i++;
                    iter.nextLineDone();
                }

            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, e.getMessage(), e);
                return;
            }
        }
    }
}
