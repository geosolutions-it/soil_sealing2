package org.geoserver.wps.area;

import java.awt.RenderingHints;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;
import java.awt.image.renderable.RenderedImageFactory;
import java.util.Set;

import javax.media.jai.ImageLayout;
import javax.media.jai.ROI;

import org.geotools.geometry.jts.ReferencedEnvelope;

import com.sun.media.jai.opimage.RIFUtil;
/**
 * {@link RenderedImageFactory} used for creating a new AreaOpImage
 *
 */
public class AreaRIF implements RenderedImageFactory {

    @Override
    public RenderedImage create(ParameterBlock paramBlock, RenderingHints hints) {
        // Selection of the source
        RenderedImage source = paramBlock.getRenderedSource(0);
        // Selection of the layout
        ImageLayout layout = RIFUtil.getImageLayoutHint(hints);
        // Selection of the parameters
        ReferencedEnvelope env = (ReferencedEnvelope) paramBlock.getObjectParameter(0);
        double multiplier = (Double) paramBlock.getObjectParameter(1);
        Set<Integer> validValues = (Set<Integer>) paramBlock.getObjectParameter(2);
        ROI roi = (ROI) paramBlock.getObjectParameter(3);
        // Creation of a new instance of the AreaOpImage
        return new AreaOpImage(source, layout, hints, env, multiplier, validValues, roi);
    }
}
