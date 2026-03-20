/**
 * GalleryScanScreen -- gallery scan UI with manual trigger, progress,
 * auto-scan toggle, permission handling, and scan results.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Switch,
} from 'react-native';
import * as MediaLibrary from 'expo-media-library';
import { Ionicons } from '@expo/vector-icons';
import { useGalleryScanStore } from '../store/useGalleryScanStore';

export default function GalleryScanScreen() {
  const {
    isScanning,
    progress,
    lastScanResult,
    error,
    scanEnabled,
    startManualScan,
    setScanEnabled,
    reset,
  } = useGalleryScanStore();

  const [permissionStatus, setPermissionStatus] = useState<string | null>(null);

  useEffect(() => {
    MediaLibrary.getPermissionsAsync().then(({ status }) => {
      setPermissionStatus(status);
    });
  }, []);

  async function handleGrantPermission() {
    const { status } = await MediaLibrary.requestPermissionsAsync();
    setPermissionStatus(status);
  }

  function handleScan() {
    startManualScan();
  }

  function handleToggleAutoScan(enabled: boolean) {
    setScanEnabled(enabled);
  }

  const needsPermission = permissionStatus !== null && permissionStatus !== 'granted';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Gallery Scan</Text>

      {/* Permission gate */}
      {needsPermission && (
        <View style={styles.card}>
          <View style={styles.permissionRow}>
            <Ionicons name="images-outline" size={24} color="#F59E0B" />
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.permissionTitle}>Gallery Access Required</Text>
              <Text style={styles.permissionDesc}>
                Tastimate needs access to your photo gallery to discover food photos.
              </Text>
            </View>
          </View>
          <Pressable style={styles.primaryBtn} onPress={handleGrantPermission}>
            <Text style={styles.primaryBtnText}>Grant Gallery Access</Text>
          </Pressable>
        </View>
      )}

      {/* Last scan result */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Scan Status</Text>
        {lastScanResult ? (
          <>
            <View style={styles.row}>
              <Text style={styles.rowLabel}>Photos classified</Text>
              <Text style={styles.rowValue}>{lastScanResult.classified}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.rowLabel}>Food photos found</Text>
              <Text style={[styles.rowValue, { color: '#16A34A', fontWeight: '700' }]}>
                {lastScanResult.foodPhotos}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.rowLabel}>Meals grouped</Text>
              <Text style={styles.rowValue}>{lastScanResult.mealGroups}</Text>
            </View>
          </>
        ) : (
          <Text style={styles.noScans}>No scans yet</Text>
        )}
      </View>

      {/* Progress indicator */}
      {isScanning && (
        <View style={styles.card}>
          <View style={styles.progressRow}>
            <ActivityIndicator size="small" color="#7C3AED" />
            <Text style={styles.progressText}>
              {progress
                ? `Classifying photo ${progress.done} of ${progress.total}...`
                : 'Starting scan...'}
            </Text>
          </View>
        </View>
      )}

      {/* Error display */}
      {error && (
        <View style={[styles.card, styles.errorCard]}>
          <Text style={styles.errorText}>{error}</Text>
          <View style={styles.errorActions}>
            <Pressable onPress={reset}>
              <Text style={styles.dismissBtn}>Dismiss</Text>
            </Pressable>
            <Pressable onPress={handleScan}>
              <Text style={styles.retryBtn}>Retry</Text>
            </Pressable>
          </View>
        </View>
      )}

      {/* Manual scan button */}
      <Pressable
        style={[styles.scanBtn, isScanning && styles.scanBtnDisabled]}
        onPress={handleScan}
        disabled={isScanning}
      >
        <Ionicons
          name="scan-outline"
          size={20}
          color={isScanning ? '#9CA3AF' : '#FFF'}
        />
        <Text style={[styles.scanBtnText, isScanning && styles.scanBtnTextDisabled]}>
          {isScanning ? 'Scanning...' : 'Scan Gallery'}
        </Text>
      </Pressable>

      {/* Auto-scan toggle */}
      <View style={styles.card}>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.toggleLabel}>Auto-scan gallery</Text>
            <Text style={styles.toggleDesc}>
              Discover food photos in the background every 4 hours. Classification
              happens when the app is open (Gemini Nano requires foreground).
            </Text>
          </View>
          <Switch
            value={scanEnabled}
            onValueChange={handleToggleAutoScan}
            trackColor={{ false: '#D1D5DB', true: '#C4B5FD' }}
            thumbColor={scanEnabled ? '#7C3AED' : '#F4F3F4'}
          />
        </View>
      </View>

      {/* Info text */}
      <View style={styles.infoCard}>
        <Ionicons name="information-circle-outline" size={18} color="#6B7280" />
        <Text style={styles.infoText}>
          Gallery scanning discovers food photos from your camera roll and classifies
          them using Gemini Nano. Auto-scan discovers photos in the background, but
          classification only runs when the app is in the foreground.
        </Text>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
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
  cardTitle: { fontSize: 18, fontWeight: '700', color: '#111827', marginBottom: 12 },

  row: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 10, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6',
  },
  rowLabel: { fontSize: 15, color: '#374151' },
  rowValue: { fontSize: 15, color: '#6B7280' },

  noScans: { fontSize: 15, color: '#9CA3AF', fontStyle: 'italic', paddingVertical: 8 },

  progressRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  progressText: { fontSize: 15, color: '#7C3AED', fontWeight: '500' },

  errorCard: { borderWidth: 1, borderColor: '#FCA5A5' },
  errorText: { fontSize: 14, color: '#DC2626', marginBottom: 12 },
  errorActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 16 },
  dismissBtn: { fontSize: 14, color: '#6B7280', fontWeight: '500' },
  retryBtn: { fontSize: 14, color: '#DC2626', fontWeight: '600' },

  scanBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    backgroundColor: '#7C3AED', borderRadius: 12, paddingVertical: 16, marginBottom: 16,
  },
  scanBtnDisabled: { backgroundColor: '#E5E7EB' },
  scanBtnText: { fontSize: 16, fontWeight: '700', color: '#FFF' },
  scanBtnTextDisabled: { color: '#9CA3AF' },

  toggleRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  toggleLabel: { fontSize: 16, fontWeight: '600', color: '#111827' },
  toggleDesc: { fontSize: 13, color: '#9CA3AF', marginTop: 4, lineHeight: 18 },

  permissionRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  permissionTitle: { fontSize: 16, fontWeight: '600', color: '#F59E0B' },
  permissionDesc: { fontSize: 13, color: '#6B7280', marginTop: 2 },
  primaryBtn: {
    backgroundColor: '#F59E0B', borderRadius: 10, paddingVertical: 12, alignItems: 'center',
  },
  primaryBtnText: { fontSize: 15, fontWeight: '700', color: '#FFF' },

  infoCard: {
    flexDirection: 'row', gap: 8, paddingHorizontal: 16, paddingVertical: 12,
  },
  infoText: { flex: 1, fontSize: 13, color: '#6B7280', lineHeight: 18 },
});
