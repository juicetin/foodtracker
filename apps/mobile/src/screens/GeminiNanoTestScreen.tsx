/**
 * GeminiNanoTestScreen -- Wave 1 spike debug screen.
 *
 * Accessible from ProfileScreen > AI Models > Gemini Nano Test.
 * Shows: availability status, image picker, Run Test button, raw JSON output, timing.
 * This is throwaway UI -- the spike evaluates Gemini Nano output quality on Pixel 9 Pro.
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { geminiNanoModule, type AvailabilityStatus } from '../../modules/gemini-nano/src/geminiNanoModule';
import { SPIKE_PROMPT } from '../services/vlm/geminiNanoService';

type TestState = 'idle' | 'running' | 'done' | 'error';

export default function GeminiNanoTestScreen() {
  const [availability, setAvailability] = useState<AvailabilityStatus | null>(null);
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [testState, setTestState] = useState<TestState>('idle');
  const [rawOutput, setRawOutput] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);

  useEffect(() => {
    geminiNanoModule
      .checkAvailability()
      .then(setAvailability)
      .catch(() => setAvailability('not_supported'));
  }, []);

  const [downloadState, setDownloadState] = useState<'idle' | 'requesting' | 'started'>('idle');

  const availabilityLabel: Record<AvailabilityStatus, string> = {
    available: 'Available',
    downloading: 'Model Downloading...',
    downloadable: 'Model Not Downloaded (tap to download)',
    unavailable: 'Not Supported on This Device',
  };

  const availabilityColor: Record<AvailabilityStatus, string> = {
    available: '#2e7d32',
    downloading: '#e65100',
    downloadable: '#f57c00',
    unavailable: '#c62828',
  };

  async function triggerDownload() {
    setDownloadState('requesting');
    try {
      const result = await geminiNanoModule.requestDownload();
      if (result === 'started' || result === 'already_available') {
        setDownloadState('started');
        // Re-poll availability so status updates
        const status = await geminiNanoModule.checkAvailability();
        setAvailability(status);
      } else {
        setDownloadState('idle');
      }
    } catch {
      setDownloadState('idle');
    }
  }

  async function pickFromCamera() {
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      setPhotoUri(result.assets[0].uri);
      setRawOutput('');
      setElapsedMs(null);
      setTestState('idle');
    }
  }

  async function pickFromGallery() {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      setPhotoUri(result.assets[0].uri);
      setRawOutput('');
      setElapsedMs(null);
      setTestState('idle');
    }
  }

  async function runTest() {
    if (!photoUri) return;
    setTestState('running');
    setRawOutput('');
    setErrorMessage('');
    setElapsedMs(null);
    const start = Date.now();
    try {
      const result = await geminiNanoModule.identifyFood(photoUri, SPIKE_PROMPT);
      setElapsedMs(Date.now() - start);
      setRawOutput(result || '(empty response)');
      setTestState('done');
    } catch (err: unknown) {
      setElapsedMs(Date.now() - start);
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
      setTestState('error');
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Gemini Nano Test</Text>
      <Text style={styles.subheading}>Wave 1 quality spike -- throwaway debug UI</Text>

      {/* Availability status */}
      <View style={styles.statusRow}>
        <Text style={styles.label}>AICore Status: </Text>
        {availability === null ? (
          <ActivityIndicator size="small" />
        ) : (
          <Text style={[styles.statusText, { color: availabilityColor[availability] }]}>
            {availabilityLabel[availability]}
          </Text>
        )}
      </View>

      {/* Download button when model not yet on device */}
      {availability === 'downloadable' && (
        <TouchableOpacity
          style={[styles.downloadButton, downloadState === 'requesting' && styles.runButtonDisabled]}
          onPress={triggerDownload}
          disabled={downloadState === 'requesting'}
        >
          {downloadState === 'requesting' ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.runButtonText}>
              {downloadState === 'started' ? 'Download Started — Check Status' : 'Download Gemini Nano Model'}
            </Text>
          )}
        </TouchableOpacity>
      )}

      {/* Image pickers */}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.button} onPress={pickFromCamera}>
          <Text style={styles.buttonText}>Camera</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={pickFromGallery}>
          <Text style={styles.buttonText}>Gallery</Text>
        </TouchableOpacity>
      </View>

      {/* Selected photo indicator */}
      {photoUri && (
        <Text style={styles.photoUri} numberOfLines={2}>
          Photo: {photoUri.split('/').pop()}
        </Text>
      )}

      {/* Run Test button */}
      <TouchableOpacity
        style={[styles.runButton, (!photoUri || testState === 'running') && styles.runButtonDisabled]}
        onPress={runTest}
        disabled={!photoUri || testState === 'running'}
      >
        {testState === 'running' ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.runButtonText}>Run Test</Text>
        )}
      </TouchableOpacity>

      {/* Timing */}
      {elapsedMs !== null && (
        <Text style={styles.timing}>{elapsedMs}ms</Text>
      )}

      {/* Error */}
      {testState === 'error' && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{errorMessage}</Text>
        </View>
      )}

      {/* Raw JSON output */}
      {testState === 'done' && (
        <View style={styles.outputBox}>
          <Text style={styles.outputLabel}>Raw JSON Response:</Text>
          <Text style={styles.outputText}>{rawOutput}</Text>
        </View>
      )}

      {/* Prompt shown for reference */}
      <View style={styles.promptBox}>
        <Text style={styles.promptLabel}>Spike Prompt:</Text>
        <Text style={styles.promptText}>{SPIKE_PROMPT}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 60 },
  heading: { fontSize: 22, fontWeight: 'bold', marginBottom: 4 },
  subheading: { fontSize: 13, color: '#888', marginBottom: 20 },
  statusRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 20 },
  label: { fontSize: 15, color: '#333' },
  statusText: { fontSize: 15, fontWeight: '600' },
  buttonRow: { flexDirection: 'row', gap: 12, marginBottom: 12 },
  button: {
    flex: 1,
    backgroundColor: '#1976d2',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontWeight: '600' },
  photoUri: { fontSize: 12, color: '#666', marginBottom: 12 },
  runButton: {
    backgroundColor: '#388e3c',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 8,
  },
  runButtonDisabled: { backgroundColor: '#aaa' },
  downloadButton: {
    backgroundColor: '#f57c00',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
  },
  runButtonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  timing: { textAlign: 'center', color: '#555', fontSize: 13, marginBottom: 12 },
  errorBox: {
    backgroundColor: '#ffebee',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: { color: '#c62828', fontFamily: 'monospace' },
  outputBox: {
    backgroundColor: '#f5f5f5',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  outputLabel: { fontWeight: '600', marginBottom: 6, color: '#333' },
  outputText: { fontFamily: 'monospace', fontSize: 13, color: '#1a1a1a' },
  promptBox: {
    backgroundColor: '#e8f5e9',
    padding: 12,
    borderRadius: 8,
    marginTop: 8,
  },
  promptLabel: { fontWeight: '600', marginBottom: 4, color: '#2e7d32' },
  promptText: { fontSize: 12, color: '#333', lineHeight: 18 },
});
