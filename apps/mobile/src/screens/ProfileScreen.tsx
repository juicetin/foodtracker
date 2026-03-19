/**
 * ProfileScreen — nutrition goals editor, preferences, AI model management.
 */

import React, { useState } from 'react';
import {
  ActivityIndicator,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../types';
import { usePreferencesStore } from '../store/usePreferencesStore';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import {
  loadExportEntries,
  loadExportRecipes,
  loadExportFavourites,
  loadExportOFFCache,
  generateCsv,
  generateJson,
} from '../services/export/exportService';

export default function ProfileScreen() {
  const rootNavigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { nutritionGoals, setNutritionGoals, region, units, setRegion, setUnits } = usePreferencesStore();

  const [editingGoals, setEditingGoals] = useState(false);
  const [calGoal, setCalGoal] = useState(String(nutritionGoals.calories));
  const [proteinGoal, setProteinGoal] = useState(String(nutritionGoals.protein));
  const [carbsGoal, setCarbsGoal] = useState(String(nutritionGoals.carbs));
  const [fatGoal, setFatGoal] = useState(String(nutritionGoals.fat));

  function saveGoals() {
    const cal = parseInt(calGoal, 10);
    const p = parseInt(proteinGoal, 10);
    const c = parseInt(carbsGoal, 10);
    const f = parseInt(fatGoal, 10);

    if ([cal, p, c, f].some((v) => isNaN(v) || v <= 0)) {
      Alert.alert('Invalid', 'All goals must be positive numbers.');
      return;
    }

    setNutritionGoals({ calories: cal, protein: p, carbs: c, fat: f });
    setEditingGoals(false);
  }

  function cancelGoalEdit() {
    setCalGoal(String(nutritionGoals.calories));
    setProteinGoal(String(nutritionGoals.protein));
    setCarbsGoal(String(nutritionGoals.carbs));
    setFatGoal(String(nutritionGoals.fat));
    setEditingGoals(false);
  }

  const regionLabels: Record<string, string> = {
    AU: 'Australia', US: 'United States', CA: 'Canada',
    UK: 'United Kingdom', FR: 'France', global: 'Global',
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Profile</Text>

      {/* Goals */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>Daily Goals</Text>
          {!editingGoals ? (
            <Pressable onPress={() => setEditingGoals(true)}>
              <Text style={styles.editBtn}>Edit</Text>
            </Pressable>
          ) : (
            <View style={styles.editActions}>
              <Pressable onPress={cancelGoalEdit}>
                <Text style={styles.cancelBtn}>Cancel</Text>
              </Pressable>
              <Pressable onPress={saveGoals}>
                <Text style={styles.saveBtn}>Save</Text>
              </Pressable>
            </View>
          )}
        </View>

        <GoalRow
          label="Calories"
          unit="kcal"
          value={calGoal}
          editing={editingGoals}
          onChange={setCalGoal}
          color="#EF4444"
        />
        <GoalRow
          label="Protein"
          unit="g"
          value={proteinGoal}
          editing={editingGoals}
          onChange={setProteinGoal}
          color="#3B82F6"
        />
        <GoalRow
          label="Carbs"
          unit="g"
          value={carbsGoal}
          editing={editingGoals}
          onChange={setCarbsGoal}
          color="#D97706"
        />
        <GoalRow
          label="Fat"
          unit="g"
          value={fatGoal}
          editing={editingGoals}
          onChange={setFatGoal}
          color="#16A34A"
        />
      </View>

      {/* Preferences */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Preferences</Text>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Region</Text>
          <Text style={styles.rowValue}>{regionLabels[region] ?? region}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Units</Text>
          <Text style={styles.rowValue}>{units === 'metric' ? 'Metric' : 'Imperial'}</Text>
        </View>
      </View>

      {/* Recipes */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recipes</Text>
        <Pressable style={styles.row} onPress={() => rootNavigation.navigate('Recipes')}>
          <Text style={styles.rowLabel}>My Recipes</Text>
          <Text style={styles.rowChevron}>→</Text>
        </Pressable>
      </View>

      {/* Export Data */}
      <ExportCard />

      {/* AI Models */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>AI Models</Text>
        <Pressable style={styles.row} onPress={() => rootNavigation.navigate('GeminiNanoTest')}>
          <Text style={styles.rowLabel}>Gemini Nano Test</Text>
          <Text style={styles.rowChevron}>→</Text>
        </Pressable>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function ExportCard() {
  const [exporting, setExporting] = useState(false);

  async function handleExport(format: 'csv' | 'json') {
    setExporting(true);
    try {
      const entries = loadExportEntries();
      const recipes = loadExportRecipes();
      const favourites = loadExportFavourites();
      const offCache = loadExportOFFCache();

      const content = format === 'csv'
        ? generateCsv(entries, recipes, favourites, offCache)
        : generateJson(entries, recipes, favourites, offCache);

      const ext = format === 'csv' ? 'csv' : 'json';
      const date = new Date().toISOString().split('T')[0];
      const filename = `tastimate-export-${date}.${ext}`;
      const fileUri = `${FileSystem.documentDirectory}${filename}`;

      await FileSystem.writeAsStringAsync(fileUri, content, {
        encoding: FileSystem.EncodingType.UTF8,
      });

      const canShare = await Sharing.isAvailableAsync();
      if (canShare) {
        await Sharing.shareAsync(fileUri, {
          mimeType: format === 'csv' ? 'text/csv' : 'application/json',
          dialogTitle: `Export ${format.toUpperCase()}`,
        });
      } else {
        Alert.alert('Saved', `Exported to ${fileUri}`);
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to export data.');
    } finally {
      setExporting(false);
    }
  }

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Export Data</Text>
      {exporting ? (
        <ActivityIndicator size="small" color="#16A34A" style={{ paddingVertical: 16 }} />
      ) : (
        <>
          <Pressable style={styles.exportBtn} onPress={() => handleExport('csv')}>
            <Ionicons name="document-text-outline" size={18} color="#16A34A" />
            <Text style={styles.exportBtnText}>Export as CSV</Text>
          </Pressable>
          <Pressable style={styles.exportBtn} onPress={() => handleExport('json')}>
            <Ionicons name="code-slash-outline" size={18} color="#3B82F6" />
            <Text style={[styles.exportBtnText, { color: '#3B82F6' }]}>Export as JSON</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

function GoalRow({ label, unit, value, editing, onChange, color }: {
  label: string; unit: string; value: string; editing: boolean;
  onChange: (v: string) => void; color: string;
}) {
  return (
    <View style={styles.goalRow}>
      <View style={styles.goalLabelRow}>
        <View style={[styles.goalDot, { backgroundColor: color }]} />
        <Text style={styles.goalLabel}>{label}</Text>
      </View>
      {editing ? (
        <View style={styles.goalInputRow}>
          <TextInput
            style={styles.goalInput}
            value={value}
            onChangeText={onChange}
            keyboardType="number-pad"
            selectTextOnFocus
          />
          <Text style={styles.goalUnit}>{unit}</Text>
        </View>
      ) : (
        <Text style={styles.goalValue}>{value} {unit}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingTop: 60, paddingHorizontal: 16 },
  title: { fontSize: 28, fontWeight: '800', color: '#111827', marginBottom: 20 },

  card: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05,
    shadowRadius: 8, elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12,
  },
  cardTitle: { fontSize: 18, fontWeight: '700', color: '#111827' },
  editBtn: { fontSize: 15, fontWeight: '600', color: '#3B82F6' },
  editActions: { flexDirection: 'row', gap: 16 },
  cancelBtn: { fontSize: 15, fontWeight: '500', color: '#6B7280' },
  saveBtn: { fontSize: 15, fontWeight: '700', color: '#16A34A' },

  goalRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  goalLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  goalDot: { width: 10, height: 10, borderRadius: 5 },
  goalLabel: { fontSize: 15, fontWeight: '500', color: '#374151' },
  goalValue: { fontSize: 15, fontWeight: '600', color: '#111827' },
  goalInputRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  goalInput: {
    backgroundColor: '#F3F4F6', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6,
    fontSize: 15, fontWeight: '600', color: '#111827', minWidth: 70, textAlign: 'right',
  },
  goalUnit: { fontSize: 13, color: '#6B7280' },

  row: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  rowLabel: { fontSize: 15, color: '#374151' },
  rowValue: { fontSize: 15, color: '#6B7280' },
  rowChevron: { fontSize: 16, color: '#D1D5DB' },
  exportBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingVertical: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  exportBtnText: { fontSize: 15, fontWeight: '500', color: '#16A34A' },
});
