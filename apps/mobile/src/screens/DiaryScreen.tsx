import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { DiaryScreenNavigationProp } from '../navigation/types';

interface DiaryScreenProps {
  navigation: DiaryScreenNavigationProp;
}

export default function DiaryScreen({ navigation }: DiaryScreenProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Food Diary</Text>
      <ScrollView style={styles.scrollView}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyText}>No entries yet</Text>
          <Text style={styles.emptySubtext}>Start tracking by adding photos on the Home tab</Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: 60,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    paddingHorizontal: 20,
    marginBottom: 16,
  },
  scrollView: {
    flex: 1,
  },
  emptyState: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
});
