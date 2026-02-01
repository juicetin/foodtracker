import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Photo } from '../types';

interface PhotoPickerProps {
  onPhotosSelected: (photos: Photo[]) => void;
  maxPhotos?: number;
}

export default function PhotoPicker({
  onPhotosSelected,
  maxPhotos = 20,
}: PhotoPickerProps) {
  const [isLoading, setIsLoading] = useState(false);

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant access to your photo library to upload food photos.',
        [{ text: 'OK' }]
      );
      return false;
    }
    return true;
  };

  const pickPhotos = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    setIsLoading(true);
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsMultipleSelection: true,
        quality: 0.8,
        selectionLimit: maxPhotos,
        exif: true,
      });

      if (!result.canceled && result.assets) {
        const photos: Photo[] = result.assets.map((asset) => ({
          id: `${Date.now()}-${Math.random()}`,
          uri: asset.uri,
          timestamp: new Date(),
          metadata: {
            width: asset.width,
            height: asset.height,
            location: asset.exif?.GPSLatitude && asset.exif?.GPSLongitude
              ? {
                  latitude: asset.exif.GPSLatitude,
                  longitude: asset.exif.GPSLongitude,
                }
              : undefined,
          },
        }));

        onPhotosSelected(photos);
      }
    } catch (error) {
      console.error('Error picking photos:', error);
      Alert.alert('Error', 'Failed to select photos. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant camera access to take photos of your food.',
        [{ text: 'OK' }]
      );
      return;
    }

    setIsLoading(true);
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.8,
        exif: true,
      });

      if (!result.canceled && result.assets) {
        const photo: Photo = {
          id: `${Date.now()}-${Math.random()}`,
          uri: result.assets[0].uri,
          timestamp: new Date(),
          metadata: {
            width: result.assets[0].width,
            height: result.assets[0].height,
            location: result.assets[0].exif?.GPSLatitude && result.assets[0].exif?.GPSLongitude
              ? {
                  latitude: result.assets[0].exif.GPSLatitude,
                  longitude: result.assets[0].exif.GPSLongitude,
                }
              : undefined,
          },
        };

        onPhotosSelected([photo]);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const showOptions = () => {
    Alert.alert(
      'Add Photos',
      'Choose how you want to add food photos',
      [
        {
          text: 'Take Photo',
          onPress: takePhoto,
        },
        {
          text: `Select from Library (up to ${maxPhotos})`,
          onPress: pickPhotos,
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ],
      { cancelable: true }
    );
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading photos...</Text>
      </View>
    );
  }

  return (
    <TouchableOpacity style={styles.button} onPress={showOptions}>
      <Text style={styles.buttonText}>Add Photos</Text>
      <Text style={styles.buttonSubtext}>Tap to select up to {maxPhotos} photos</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  buttonSubtext: {
    color: '#fff',
    fontSize: 12,
    opacity: 0.8,
  },
  loadingContainer: {
    padding: 32,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
});
