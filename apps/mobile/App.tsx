// Register background tasks at module load (must run before React renders)
import './src/services/backup/backupScheduler';
import './src/services/gallery/galleryScanScheduler';

import { useEffect, useRef } from 'react';
import { AppState, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { RootNavigator } from './src/navigation';
import { usePreferencesStore } from './src/store/usePreferencesStore';
import { triggerForegroundDrain } from './src/services/gallery/galleryScanScheduler';

export default function App() {
  const initRegion = usePreferencesStore((s) => s.initRegionFromLocale);
  useEffect(() => { initRegion(); }, [initRegion]);

  // Silently drain gallery scan queue when app comes to foreground
  const appState = useRef(AppState.currentState);
  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      if (appState.current.match(/inactive|background/) && nextState === 'active') {
        // Fire-and-forget: drain queue silently in background
        triggerForegroundDrain().catch(() => {
          // Silent -- gallery drain is opportunistic
        });
      }
      appState.current = nextState;
    });
    return () => sub.remove();
  }, []);

  return (
    <GestureHandlerRootView style={styles.container}>
      <RootNavigator />
      <StatusBar style="auto" />
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
