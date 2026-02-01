import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import { HomeScreenNavigationProp } from '../navigation/types';
import { PhotoPicker, BatchPhotoGrid } from '../components';
import { useFoodLogStore } from '../store';
import { Photo } from '../types';

interface HomeScreenProps {
  navigation: HomeScreenNavigationProp;
}

export default function HomeScreen({ navigation }: HomeScreenProps) {
  const {
    selectedPhotos,
    setSelectedPhotos,
    clearSelectedPhotos,
    isProcessing,
    setIsProcessing,
    getTodayTotals,
  } = useFoodLogStore();

  const todayTotals = getTodayTotals();

  const handlePhotosSelected = (photos: Photo[]) => {
    setSelectedPhotos([...selectedPhotos, ...photos]);
  };

  const handleRemovePhoto = (photoId: string) => {
    setSelectedPhotos(selectedPhotos.filter(p => p.id !== photoId));
  };

  const handleProcessPhotos = async () => {
    if (selectedPhotos.length === 0) {
      Alert.alert('No Photos', 'Please select photos first.');
      return;
    }

    setIsProcessing(true);

    // TODO: Call AI service to process photos
    Alert.alert(
      'Processing Photos',
      `Processing ${selectedPhotos.length} photo${selectedPhotos.length !== 1 ? 's' : ''}...\n\nThis will connect to the AI service to analyze your food.`,
      [
        {
          text: 'OK',
          onPress: () => {
            // Simulate processing
            setTimeout(() => {
              setIsProcessing(false);
              Alert.alert('Success', 'Photos processed! (Demo mode)');
              clearSelectedPhotos();
            }, 2000);
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Food Tracker</Text>
        <Text style={styles.subtitle}>Track your meals with AI-powered photo analysis</Text>

        <PhotoPicker onPhotosSelected={handlePhotosSelected} maxPhotos={20} />

        {selectedPhotos.length > 0 && (
          <>
            <BatchPhotoGrid
              photos={selectedPhotos}
              onRemovePhoto={handleRemovePhoto}
            />

            <View style={styles.actionButtons}>
              <TouchableOpacity
                style={[styles.secondaryButton, { flex: 1, marginRight: 8 }]}
                onPress={clearSelectedPhotos}
              >
                <Text style={styles.secondaryButtonText}>Clear All</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.primaryButton, { flex: 2 }]}
                onPress={handleProcessPhotos}
                disabled={isProcessing}
              >
                <Text style={styles.primaryButtonText}>
                  {isProcessing ? 'Processing...' : `Process ${selectedPhotos.length} Photo${selectedPhotos.length !== 1 ? 's' : ''}`}
                </Text>
              </TouchableOpacity>
            </View>
          </>
        )}

        <View style={styles.statsSection}>
          <Text style={styles.statsTitle}>Today's Totals</Text>
          <View style={styles.statsContainer}>
            <View style={styles.statBox}>
              <Text style={styles.statValue}>{Math.round(todayTotals.calories)}</Text>
              <Text style={styles.statLabel}>Calories</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statValue}>{Math.round(todayTotals.protein)}g</Text>
              <Text style={styles.statLabel}>Protein</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statValue}>{Math.round(todayTotals.carbs)}g</Text>
              <Text style={styles.statLabel}>Carbs</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statValue}>{Math.round(todayTotals.fat)}g</Text>
              <Text style={styles.statLabel}>Fat</Text>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    padding: 20,
    paddingTop: 60,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 24,
  },
  actionButtons: {
    flexDirection: 'row',
    marginTop: 16,
    marginBottom: 32,
  },
  primaryButton: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: '#F5F5F5',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#333',
    fontSize: 16,
    fontWeight: '600',
  },
  statsSection: {
    marginTop: 16,
  },
  statsTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statBox: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    padding: 16,
    borderRadius: 12,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
});
