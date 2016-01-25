/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.geoserver.catalog.FeatureTypeInfo;
import org.geotools.data.DefaultTransaction;
import org.geotools.data.FeatureReader;
import org.geotools.data.Query;
import org.geotools.data.Transaction;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.jdbc.JDBCDataStore;
import org.opengis.feature.Feature;
import org.opengis.feature.Property;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.PrecisionModel;

/**
 * @author Alessio Fabiani, GeoSolutions S.A.S.
 * 
 */
public class SoilSealingAdministrativeUnit {

    /**
     * Geometry and Filter Factories
     */
    private static final FilterFactory ff = CommonFactoryFinder.getFilterFactory(null);

    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory(
            new PrecisionModel());

    /**
     * GeoCoding reference layers
     */
    private FeatureTypeInfo geoCodingReference;

    private FeatureTypeInfo populationReference;

    private String name;

    private String parent;

    private AuType type;

    private Geometry the_geom;

    private Map<String, Integer> population = new HashMap<String, Integer>();

    private List<SoilSealingAdministrativeUnit> subs = new LinkedList<SoilSealingAdministrativeUnit>();

    /**
     * Default constructor
     * 
     * @param au
     * @param geoCodingReference
     * @param populationReference
     * @throws IOException
     */
    public SoilSealingAdministrativeUnit(String au, FeatureTypeInfo geoCodingReference,
            FeatureTypeInfo populationReference) throws IOException {
        if (!au.contains("_")) {
            throw new IOException("Invalid Administrative Unit name");
        }
        this.geoCodingReference = geoCodingReference;
        this.populationReference = populationReference;
        this.name = au.split("_")[0];
        this.parent = au.split("_")[1];

        FeatureReader<SimpleFeatureType, SimpleFeature> ftReader = null;
        Transaction transaction = new DefaultTransaction();
        try {
            final JDBCDataStore ds = (JDBCDataStore) geoCodingReference.getStore().getDataStore(null);
            Filter nameFilter = ff.equals(ff.property("name"), ff.literal(this.name));
            Filter parentFilter = ff.equals(ff.property("parent"), ff.literal(this.parent));
            Filter queryFilter = ff.and(Arrays.asList(nameFilter, parentFilter));
            Query query = new Query(geoCodingReference.getFeatureType().getName().getLocalPart(), queryFilter);
            
            ftReader = ds.getFeatureReader(query, transaction);

            while (ftReader.hasNext()) {
                Feature feature = ftReader.next();
                this.type = AuType.getType((Integer) feature.getProperty("type").getValue());
                this.the_geom = (Geometry) feature.getDefaultGeometryProperty().getValue();
                break;
            }
        } finally {
            if (ftReader != null) {
                ftReader.close();
            }

            transaction.commit();
            transaction.close();
        }

        if (this.type == null || this.the_geom == null) {
            throw new IOException("Invalid Administrative Unit name: no record found!");
        }

        switch (this.type) {
        case MUNICIPALITY:
            loadPopulationStatistics(this);
            break;
        case DISTRICT:
        case REGION:
            loadSubs(this);
            break;
        default:
            break;
        }
    }

    /**
     * 
     * @param soilSealingAdministrativeUnit
     * @throws IOException
     */
    private void loadSubs(SoilSealingAdministrativeUnit soilSealingAdministrativeUnit) throws IOException {
        FeatureReader<SimpleFeatureType, SimpleFeature> ftReader = null;
        Transaction transaction = new DefaultTransaction();
        try {
            final JDBCDataStore ds = (JDBCDataStore) geoCodingReference.getStore().getDataStore(null);
            Filter parentFilter = ff.equals(ff.property("parent"), ff.literal(this.name));
            Filter typeFilter = ff.equals(ff.property("type"), ff.literal(this.type.getValue() + 1));
            Filter queryFilter = ff.and(Arrays.asList(typeFilter, parentFilter));
            Query query = new Query(geoCodingReference.getFeatureType().getName().getLocalPart(), queryFilter);

            ftReader = ds.getFeatureReader(query, transaction);

            List<String> ftSubs = new ArrayList<String>();
            while (ftReader.hasNext()) {
                Feature feature = ftReader.next();
                String au = (String) feature.getProperty("name").getValue() + "_" + this.name;
                ftSubs.add(au);
            }

            for (String au : ftSubs) {
                this.subs.add(new SoilSealingAdministrativeUnit(au, geoCodingReference,
                        populationReference));
            }
        } finally {
            if (ftReader != null) {
                ftReader.close();
            }

            transaction.commit();
            transaction.close();
        }
    }

    /**
     * 
     * @param soilSealingAdministrativeUnit
     * @throws IOException
     */
    private void loadPopulationStatistics(SoilSealingAdministrativeUnit soilSealingAdministrativeUnit) throws IOException {
        FeatureReader<SimpleFeatureType, SimpleFeature> ftReader = null;
        Transaction transaction = new DefaultTransaction();
        try {
            final JDBCDataStore ds = (JDBCDataStore) populationReference.getStore().getDataStore(null);
            Filter auNameFilter = ff.equals(ff.property("au_name"), ff.literal(this.name));
            Filter queryFilter = ff.and(Arrays.asList(auNameFilter));
            Query query = new Query(populationReference.getFeatureType().getName().getLocalPart(), queryFilter);

            ftReader = ds.getFeatureReader(query, transaction);
            
            while (ftReader.hasNext()) {
                Feature feature = ftReader.next();
                Collection<Property> properties = feature.getProperties();
                for (Property prop : properties) {
                    if (prop.getName().getLocalPart().startsWith("a_")) {
                        Object yearPopulationValue = prop.getValue();
                        if (yearPopulationValue != null) {
                            population.put(prop.getName().getLocalPart().split("a_")[1], ((BigDecimal) yearPopulationValue).intValue());
                        }
                    }
                }
            }
        } finally {
            if (ftReader != null) {
                ftReader.close();
            }

            transaction.commit();
            transaction.close();
        }
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @return the parent
     */
    public String getParent() {
        return parent;
    }

    /**
     * @return the type
     */
    public AuType getType() {
        return type;
    }

    /**
     * @return the the_geom
     */
    public Geometry getTheGeom() {
        return the_geom;
    }

    /**
     * @return the population
     */
    public Map<String, Integer> getPopulation() {
        return population;
    }

    /**
     * @return the subs
     */
    public List<SoilSealingAdministrativeUnit> getSubs() {
        return subs;
    }

    /**
     * {@linkplain AuSelectionType} enum
     * 
     * @author Alessio
     * 
     */
    public enum AuSelectionType {
        AU_LIST, AU_SUBS
    }

    /**
     * {@linkplain AuType} enum
     * 
     * @author Alessio
     * 
     */
    public enum AuType {
        MUNICIPALITY(2), DISTRICT(1), REGION(0);

        private int value;

        private AuType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }

        public static AuType getType(int value) {
            for (AuType auType : AuType.values()) {
                if (auType.value == value)
                    return auType;
            }

            return null;
        }

    }
}
