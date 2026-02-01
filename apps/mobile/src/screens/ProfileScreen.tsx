import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { ProfileScreenNavigationProp } from '../navigation/types';

interface ProfileScreenProps {
  navigation: ProfileScreenNavigationProp;
}

export default function ProfileScreen({ navigation }: ProfileScreenProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Profile</Text>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Goals</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Daily Calories</Text>
          <Text style={styles.value}>2000 kcal</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Protein</Text>
          <Text style={styles.value}>150g</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Carbs</Text>
          <Text style={styles.value}>200g</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Fat</Text>
          <Text style={styles.value}>65g</Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Preferences</Text>
        <TouchableOpacity style={styles.row}>
          <Text style={styles.label}>Region</Text>
          <Text style={styles.value}>Australia</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.row}>
          <Text style={styles.label}>Units</Text>
          <Text style={styles.value}>Metric</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 24,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  label: {
    fontSize: 16,
    color: '#333',
  },
  value: {
    fontSize: 16,
    color: '#666',
  },
});
