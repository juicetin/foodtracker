import React from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Photo } from '../types';

interface BatchPhotoGridProps {
  photos: Photo[];
  onRemovePhoto?: (photoId: string) => void;
  onPhotoPress?: (photo: Photo) => void;
}

const SCREEN_WIDTH = Dimensions.get('window').width;
const GRID_GAP = 8;
const PHOTOS_PER_ROW = 3;
const PHOTO_SIZE = (SCREEN_WIDTH - (GRID_GAP * (PHOTOS_PER_ROW + 1))) / PHOTOS_PER_ROW;

export default function BatchPhotoGrid({
  photos,
  onRemovePhoto,
  onPhotoPress,
}: BatchPhotoGridProps) {
  if (photos.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Selected Photos</Text>
        <Text style={styles.count}>{photos.length} photo{photos.length !== 1 ? 's' : ''}</Text>
      </View>

      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {photos.map((photo, index) => (
          <View key={photo.id} style={styles.photoContainer}>
            <TouchableOpacity
              onPress={() => onPhotoPress?.(photo)}
              activeOpacity={0.7}
            >
              <Image
                source={{ uri: photo.uri }}
                style={styles.photo}
                resizeMode="cover"
              />
              <View style={styles.photoOverlay}>
                <Text style={styles.photoNumber}>{index + 1}</Text>
              </View>
            </TouchableOpacity>

            {onRemovePhoto && (
              <TouchableOpacity
                style={styles.removeButton}
                onPress={() => onRemovePhoto(photo.id)}
              >
                <Text style={styles.removeButtonText}>×</Text>
              </TouchableOpacity>
            )}
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    paddingHorizontal: 4,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  count: {
    fontSize: 14,
    color: '#666',
  },
  scrollContent: {
    paddingHorizontal: 4,
  },
  photoContainer: {
    marginRight: 12,
    position: 'relative',
  },
  photo: {
    width: 120,
    height: 120,
    borderRadius: 12,
    backgroundColor: '#F5F5F5',
  },
  photoOverlay: {
    position: 'absolute',
    top: 8,
    left: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  photoNumber: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  removeButton: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: '#FF3B30',
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  removeButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    lineHeight: 22,
  },
});
