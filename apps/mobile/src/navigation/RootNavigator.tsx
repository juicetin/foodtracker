import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import MainTabNavigator from './MainTabNavigator';
import { DetectionScreen, VlmDownloadScreen, GeminiNanoTestScreen, EntryDetailScreen, FoodSearchScreen, BarcodeScanScreen, RecipeScreen, QuickAddScreen, ReidentifyMergeScreen, GalleryScanScreen, ScaleInputScreen, WeightTrendScreen } from '../screens';
import SyncSettingsScreen from '../screens/SyncSettingsScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="Main" component={MainTabNavigator} />
        <Stack.Screen
          name="Detection"
          component={DetectionScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
          }}
        />
        <Stack.Screen
          name="VlmDownload"
          component={VlmDownloadScreen}
          options={{
            headerShown: true,
            headerTitle: 'VLM Model',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="GeminiNanoTest"
          component={GeminiNanoTestScreen}
          options={{
            headerShown: true,
            headerTitle: 'Gemini Nano Test',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="FoodSearch"
          component={FoodSearchScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="BarcodeScan"
          component={BarcodeScanScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="Recipes"
          component={RecipeScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="QuickAdd"
          component={QuickAddScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="EntryDetail"
          component={EntryDetailScreen}
          options={{
            headerShown: true,
            headerTitle: 'Meal Detail',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="SyncSettings"
          component={SyncSettingsScreen}
          options={{
            headerShown: true,
            headerTitle: 'Google Drive Sync',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="ReidentifyMerge"
          component={ReidentifyMergeScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="GalleryScan"
          component={GalleryScanScreen}
          options={{
            headerShown: true,
            headerTitle: 'Gallery Scan',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="ScaleInput"
          component={ScaleInputScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="WeightTrend"
          component={WeightTrendScreen}
          options={{
            animation: 'slide_from_right',
            headerShown: false,
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
