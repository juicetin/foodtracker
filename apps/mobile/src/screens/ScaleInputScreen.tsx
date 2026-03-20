/**
 * ScaleInputScreen -- Shows OCR scale reading with manual override and container tare selection.
 *
 * Navigation params:
 *   - photoUri (optional): If provided, runs OCR on mount.
 *   - onResult (optional): Callback receiving the confirmed net weight in grams.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { RootStackParamList } from '../types';
import { readScaleWeight, type ScaleReading } from '../services/scale/scaleOcrService';
import {
  addContainer,
  applyTare,
  deleteContainer,
  getContainers,
  recordContainerUsage,
  type Container,
} from '../services/scale/containerService';

type ScaleInputRoute = RouteProp<RootStackParamList, 'ScaleInput'>;

export default function ScaleInputScreen() {
  const navigation = useNavigation();
  const route = useRoute<ScaleInputRoute>();
  const photoUri = route.params?.photoUri ?? null;

  // Scale reading state
  const [reading, setReading] = useState<ScaleReading | null>(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [manualWeight, setManualWeight] = useState('');

  // Container state
  const [containers, setContainers] = useState<Container[]>([]);
  const [selectedContainer, setSelectedContainer] = useState<Container | null>(null);

  // Add container form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newContainerName, setNewContainerName] = useState('');
  const [newContainerWeight, setNewContainerWeight] = useState('');

  // Derived: gross weight
  const grossWeight = parseFloat(manualWeight) || 0;
  const netWeight = selectedContainer
    ? applyTare(grossWeight, selectedContainer)
    : grossWeight;

  // Load containers on mount
  useEffect(() => {
    loadContainers();
  }, []);

  // Run OCR if photoUri provided
  useEffect(() => {
    if (photoUri) {
      runOcr(photoUri);
    }
  }, [photoUri]);

  async function loadContainers() {
    try {
      const list = await getContainers();
      setContainers(list);
    } catch {
      // Silently handle -- containers are optional
    }
  }

  async function runOcr(uri: string) {
    setOcrLoading(true);
    try {
      const result = await readScaleWeight(uri);
      setReading(result);
      if (result) {
        setManualWeight(String(Math.round(result.weightG * 10) / 10));
      }
    } catch {
      // OCR failed -- user can enter manually
    } finally {
      setOcrLoading(false);
    }
  }

  function handleContainerSelect(container: Container) {
    if (selectedContainer?.id === container.id) {
      setSelectedContainer(null);
    } else {
      setSelectedContainer(container);
    }
  }

  async function handleAddContainer() {
    const name = newContainerName.trim();
    const weight = parseFloat(newContainerWeight);
    if (!name) {
      Alert.alert('Error', 'Container name is required.');
      return;
    }
    if (isNaN(weight) || weight <= 0) {
      Alert.alert('Error', 'Enter a valid weight in grams.');
      return;
    }
    try {
      await addContainer(name, weight);
      setNewContainerName('');
      setNewContainerWeight('');
      setShowAddForm(false);
      await loadContainers();
    } catch {
      Alert.alert('Error', 'Failed to add container.');
    }
  }

  async function handleDeleteContainer(id: number) {
    try {
      await deleteContainer(id);
      if (selectedContainer?.id === id) {
        setSelectedContainer(null);
      }
      await loadContainers();
    } catch {
      Alert.alert('Error', 'Failed to delete container.');
    }
  }

  async function handleConfirm() {
    if (netWeight <= 0) {
      Alert.alert('Invalid Weight', 'Please enter a weight greater than 0.');
      return;
    }
    // Record container usage if selected
    if (selectedContainer) {
      try {
        await recordContainerUsage(selectedContainer.id);
      } catch {
        // Non-critical
      }
    }
    // Navigate back with result
    if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => navigation.goBack()} style={styles.headerBack}>
          <Text style={styles.headerBackText}>Cancel</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Scale Weight</Text>
        <View style={styles.headerRight} />
      </View>

      <View style={styles.body}>
        {/* OCR Result */}
        {ocrLoading ? (
          <View style={styles.ocrSection}>
            <ActivityIndicator size="small" color="#16A34A" />
            <Text style={styles.ocrLoadingText}>Reading scale...</Text>
          </View>
        ) : reading ? (
          <View style={styles.ocrSection}>
            <View style={styles.ocrResult}>
              <Text style={styles.ocrLabel}>Scale Reading</Text>
              <View style={styles.ocrValueRow}>
                <Text style={styles.ocrValue}>
                  {Math.round(reading.weightG * 10) / 10}g
                </Text>
                <View
                  style={[
                    styles.confidenceBadge,
                    {
                      backgroundColor:
                        reading.confidence === 'high' ? '#DCFCE7' : '#FEF9C3',
                    },
                  ]}
                >
                  <Text
                    style={[
                      styles.confidenceText,
                      {
                        color:
                          reading.confidence === 'high' ? '#16A34A' : '#CA8A04',
                      },
                    ]}
                  >
                    {reading.confidence === 'high' ? 'High confidence' : 'Low confidence'}
                  </Text>
                </View>
              </View>
            </View>
          </View>
        ) : photoUri ? (
          <View style={styles.ocrSection}>
            <Text style={styles.ocrFallbackText}>
              Could not read scale. Enter weight manually.
            </Text>
          </View>
        ) : null}

        {/* Manual Weight Input */}
        <View style={styles.inputSection}>
          <Text style={styles.sectionTitle}>Weight (grams)</Text>
          <TextInput
            style={styles.weightInput}
            value={manualWeight}
            onChangeText={setManualWeight}
            keyboardType="decimal-pad"
            placeholder="0.0"
            placeholderTextColor="#D1D5DB"
          />
        </View>

        {/* Container Selector */}
        <View style={styles.containerSection}>
          <View style={styles.containerHeader}>
            <Text style={styles.sectionTitle}>Container Tare</Text>
            <Pressable onPress={() => setShowAddForm(!showAddForm)}>
              <Text style={styles.addContainerBtn}>
                {showAddForm ? 'Cancel' : '+ Add'}
              </Text>
            </Pressable>
          </View>

          {showAddForm && (
            <View style={styles.addForm}>
              <TextInput
                style={styles.addFormInput}
                value={newContainerName}
                onChangeText={setNewContainerName}
                placeholder="Container name"
                placeholderTextColor="#9CA3AF"
              />
              <TextInput
                style={[styles.addFormInput, { width: 100 }]}
                value={newContainerWeight}
                onChangeText={setNewContainerWeight}
                keyboardType="decimal-pad"
                placeholder="Weight (g)"
                placeholderTextColor="#9CA3AF"
              />
              <Pressable style={styles.addFormSave} onPress={handleAddContainer}>
                <Text style={styles.addFormSaveText}>Save</Text>
              </Pressable>
            </View>
          )}

          {containers.length === 0 ? (
            <Text style={styles.emptyText}>No containers. Add one to subtract tare weight.</Text>
          ) : (
            <FlatList
              data={containers}
              horizontal
              showsHorizontalScrollIndicator={false}
              keyExtractor={(item) => String(item.id)}
              contentContainerStyle={styles.containerList}
              renderItem={({ item }) => {
                const isSelected = selectedContainer?.id === item.id;
                return (
                  <Pressable
                    style={[
                      styles.containerPill,
                      isSelected && styles.containerPillSelected,
                    ]}
                    onPress={() => handleContainerSelect(item)}
                    onLongPress={() => {
                      Alert.alert(
                        'Delete Container',
                        `Delete "${item.name}"?`,
                        [
                          { text: 'Cancel', style: 'cancel' },
                          {
                            text: 'Delete',
                            style: 'destructive',
                            onPress: () => handleDeleteContainer(item.id),
                          },
                        ],
                      );
                    }}
                  >
                    <Text
                      style={[
                        styles.containerPillName,
                        isSelected && styles.containerPillTextSelected,
                      ]}
                    >
                      {item.name}
                    </Text>
                    <Text
                      style={[
                        styles.containerPillWeight,
                        isSelected && styles.containerPillTextSelected,
                      ]}
                    >
                      {item.weightGrams}g
                    </Text>
                  </Pressable>
                );
              }}
            />
          )}
        </View>

        {/* Net Weight Display */}
        <View style={styles.netWeightSection}>
          <Text style={styles.netWeightLabel}>Net Weight</Text>
          <Text style={styles.netWeightValue}>{Math.round(netWeight * 10) / 10}g</Text>
          {selectedContainer && (
            <Text style={styles.netWeightCalc}>
              {grossWeight}g - {selectedContainer.weightGrams}g ({selectedContainer.name})
            </Text>
          )}
        </View>
      </View>

      {/* Confirm Button */}
      <View style={styles.footer}>
        <Pressable
          style={[styles.confirmBtn, netWeight <= 0 && styles.confirmBtnDisabled]}
          onPress={handleConfirm}
          disabled={netWeight <= 0}
        >
          <Text style={styles.confirmBtnText}>
            Confirm {Math.round(netWeight * 10) / 10}g
          </Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#FFFFFF', paddingHorizontal: 16, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E5E7EB',
  },
  headerBack: { padding: 4 },
  headerBackText: { fontSize: 15, color: '#3B82F6', fontWeight: '500' },
  headerTitle: { fontSize: 17, fontWeight: '700', color: '#111827' },
  headerRight: { width: 60 },

  body: { flex: 1, paddingHorizontal: 16, paddingTop: 16 },

  ocrSection: {
    backgroundColor: '#FFFFFF', borderRadius: 16, padding: 16, marginBottom: 16,
    flexDirection: 'row', alignItems: 'center', gap: 12,
  },
  ocrLoadingText: { fontSize: 14, color: '#6B7280' },
  ocrResult: { flex: 1 },
  ocrLabel: { fontSize: 13, color: '#6B7280', marginBottom: 4 },
  ocrValueRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  ocrValue: { fontSize: 28, fontWeight: '800', color: '#111827' },
  confidenceBadge: { borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  confidenceText: { fontSize: 12, fontWeight: '600' },
  ocrFallbackText: { fontSize: 14, color: '#6B7280', fontStyle: 'italic' },

  inputSection: {
    backgroundColor: '#FFFFFF', borderRadius: 16, padding: 16, marginBottom: 16,
  },
  sectionTitle: { fontSize: 15, fontWeight: '600', color: '#374151', marginBottom: 8 },
  weightInput: {
    fontSize: 32, fontWeight: '800', color: '#111827',
    paddingVertical: 8, borderBottomWidth: 2, borderBottomColor: '#16A34A',
  },

  containerSection: {
    backgroundColor: '#FFFFFF', borderRadius: 16, padding: 16, marginBottom: 16,
  },
  containerHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12,
  },
  addContainerBtn: { fontSize: 14, fontWeight: '600', color: '#3B82F6' },
  addForm: {
    flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 12,
  },
  addFormInput: {
    flex: 1, backgroundColor: '#F3F4F6', borderRadius: 8,
    paddingHorizontal: 12, paddingVertical: 8, fontSize: 14, color: '#111827',
  },
  addFormSave: {
    backgroundColor: '#16A34A', borderRadius: 8, paddingHorizontal: 14, paddingVertical: 8,
  },
  addFormSaveText: { fontSize: 14, fontWeight: '600', color: '#FFFFFF' },
  containerList: { gap: 8 },
  containerPill: {
    backgroundColor: '#F3F4F6', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 10,
    alignItems: 'center', borderWidth: 1.5, borderColor: 'transparent',
  },
  containerPillSelected: {
    borderColor: '#16A34A', backgroundColor: '#F0FDF4',
  },
  containerPillName: { fontSize: 13, fontWeight: '600', color: '#374151' },
  containerPillWeight: { fontSize: 12, color: '#6B7280', marginTop: 2 },
  containerPillTextSelected: { color: '#16A34A' },
  emptyText: { fontSize: 13, color: '#9CA3AF', textAlign: 'center', paddingVertical: 12 },

  netWeightSection: {
    backgroundColor: '#FFFFFF', borderRadius: 16, padding: 16, alignItems: 'center',
  },
  netWeightLabel: { fontSize: 13, color: '#6B7280', marginBottom: 4 },
  netWeightValue: { fontSize: 40, fontWeight: '800', color: '#16A34A' },
  netWeightCalc: { fontSize: 12, color: '#9CA3AF', marginTop: 4 },

  footer: {
    backgroundColor: '#FFFFFF', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 28,
    borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: '#E5E7EB',
  },
  confirmBtn: {
    backgroundColor: '#16A34A', borderRadius: 14, paddingVertical: 16, alignItems: 'center',
  },
  confirmBtnDisabled: { backgroundColor: '#D1D5DB' },
  confirmBtnText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700', letterSpacing: 0.3 },
});
