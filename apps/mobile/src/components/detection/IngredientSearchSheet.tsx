/**
 * Ingredient search modal — shown when user taps an ingredient name.
 *
 * Queries the KG for dishes and recipe ingredients matching the search text.
 * Shows closest matches first, filters as user types.
 * Falls back to free-text entry if KG is unavailable.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { getKnowledgeGraphService } from '../../services/knowledge-graph';
import { searchProducts } from '../../services/openfoodfacts/openFoodFactsService';

interface Props {
  visible: boolean;
  initialQuery: string;
  onSelect: (ingredientName: string) => void;
  onDismiss: () => void;
}

export default function IngredientSearchSheet({
  visible,
  initialQuery,
  onSelect,
  onDismiss,
}: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState<{ name: string; source: 'kg' | 'off' }[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<TextInput>(null);

  // Reset state when opened
  useEffect(() => {
    if (visible) {
      setQuery(initialQuery);
      doSearch(initialQuery);
      // Focus input after modal animation
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [visible, initialQuery]);

  const doSearch = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      setResults([]);
      return;
    }

    setLoading(true);
    try {
      const unified: { name: string; source: 'kg' | 'off' }[] = [];

      // KG search (local, fast)
      const kg = await getKnowledgeGraphService();
      if (kg) {
        const matches = await kg.searchIngredients(trimmed, 15);
        for (const name of matches) {
          unified.push({ name, source: 'kg' });
        }
      }

      // OFF search (remote, broader coverage)
      const offResults = await searchProducts(trimmed, 8);
      const kgNames = new Set(unified.map((r) => r.name.toLowerCase()));
      for (const p of offResults) {
        const displayName = p.brand ? `${p.name} (${p.brand})` : p.name;
        if (!kgNames.has(p.name.toLowerCase())) {
          unified.push({ name: displayName, source: 'off' });
        }
      }

      setResults(unified);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  function handleTextChange(text: string) {
    setQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(text), 250);
  }

  function handleSelectResult(item: { name: string; source: string }) {
    onSelect(item.name);
  }

  function handleSubmitCustom() {
    const trimmed = query.trim();
    if (trimmed) onSelect(trimmed);
  }

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent
      onRequestClose={onDismiss}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.overlay}
      >
        <Pressable style={styles.backdrop} onPress={onDismiss} />
        <View style={styles.sheet}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Change Ingredient</Text>
            <Pressable onPress={onDismiss} hitSlop={12}>
              <Text style={styles.headerClose}>Cancel</Text>
            </Pressable>
          </View>

          {/* Search input */}
          <View style={styles.searchRow}>
            <TextInput
              ref={inputRef}
              style={styles.searchInput}
              value={query}
              onChangeText={handleTextChange}
              onSubmitEditing={handleSubmitCustom}
              placeholder="Search ingredients..."
              placeholderTextColor="#9CA3AF"
              returnKeyType="done"
              autoCorrect={false}
              autoCapitalize="none"
            />
            {query.trim().length > 0 && (
              <Pressable
                style={styles.useCustomBtn}
                onPress={handleSubmitCustom}
              >
                <Text style={styles.useCustomText}>Use "{query.trim()}"</Text>
              </Pressable>
            )}
          </View>

          {/* Results */}
          <FlatList
            data={results}
            keyExtractor={(item, index) => `${item.name}-${index}`}
            keyboardShouldPersistTaps="handled"
            style={styles.list}
            renderItem={({ item }) => (
              <Pressable
                style={styles.resultRow}
                onPress={() => handleSelectResult(item)}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Text style={[styles.resultText, { flex: 1 }]} numberOfLines={1}>{item.name}</Text>
                  <Text style={styles.sourceTag}>{item.source === 'off' ? 'OFF' : 'KG'}</Text>
                </View>
              </Pressable>
            )}
            ListEmptyComponent={
              !loading ? (
                <Text style={styles.emptyText}>
                  {query.trim().length > 0
                    ? 'No matches found. Tap "Use" above to enter custom name.'
                    : 'Type to search the food database'}
                </Text>
              ) : null
            }
          />
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  backdrop: {
    flex: 1,
  },
  sheet: {
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '75%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 10,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#F3F4F6',
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#111827',
  },
  headerClose: {
    fontSize: 15,
    color: '#6B7280',
    fontWeight: '500',
  },
  searchRow: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
  },
  searchInput: {
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    color: '#111827',
  },
  useCustomBtn: {
    backgroundColor: '#F0FDF4',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    alignSelf: 'flex-start',
  },
  useCustomText: {
    fontSize: 13,
    color: '#16A34A',
    fontWeight: '600',
  },
  list: {
    paddingBottom: 40,
  },
  resultRow: {
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#F9FAFB',
  },
  resultText: {
    fontSize: 15,
    color: '#111827',
  },
  sourceTag: {
    fontSize: 10,
    fontWeight: '700',
    color: '#9CA3AF',
    backgroundColor: '#F3F4F6',
    borderRadius: 4,
    paddingHorizontal: 5,
    paddingVertical: 2,
    marginLeft: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    paddingVertical: 24,
    paddingHorizontal: 20,
  },
});
