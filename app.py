        if not blind_bin_id.strip():
            st.error("Enter a bin ID.")

        elif blind_photo is None:
            st.error("Upload a bin photograph.")

        else:
            try:
                image = Image.open(blind_photo).convert("RGB")

                start = time.perf_counter()
                predictions = predict_quantities(image)
                elapsed = time.perf_counter() - start

                write_log(
                    {},
                    f"blind:{blind_bin_id.strip()}:{blind_photo.name}",
                    predictions,
                    "BLIND_PREDICTION",
                    elapsed,
                )

                st.image(
                    image,
                    caption="Blind bin photograph",
                    width=320,
                )

                st.success(
                    f"Predictions generated for bin {blind_bin_id.strip()}."
                )

                prediction_df = pd.DataFrame([
                    {
                        "Product ASIN": asin,
                        "Predicted quantity": predictions[asin],
                    }
                    for asin in model_asins
                ])

                st.dataframe(
                    prediction_df,
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download blind predictions",
                    data=prediction_df.to_csv(index=False),
                    file_name="blind_predictions.csv",
                    mime="text/csv",
                    key="download_blind_predictions",
                )

                st.caption(
                    "Predictions are model estimates, not ground-truth labels. "
                    f"Inference time: {elapsed:.4f} seconds."
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                st.error(
                    f"The selected file is not a readable image: {error}"
                )

            except Exception as error:
                st.error(f"Blind prediction failed: {error}")


# ==================================================
# TAB 3: ABOUT THE MODEL
# ==================================================

with tab_about:
    st.subheader("Model information")

    st.write(f"Products in model catalogue: **{len(model_asins)}**")
    st.write(f"Maximum predicted quantity per product: **{max_quantity}**")
    st.write(f"Compute device: **{DEVICE}**")

    st.info(
        "This prototype only covers the ten selected catalogue products. "
        "A predicted quantity of zero does not prove an item is absent. "
        "The model can make false positives and false negatives, so it "
        "should not be treated as a production-grade guarantee."
    )

    st.markdown("#### Data handling")

    st.write(
        "Order validation uses model predictions from the selected image "
        "and the quantities entered by the user. The app does not load "
        "blind-bin metadata to make predictions. Demo mode reads images "
        "only from the dedicated data/demo_images folder."
    )

    st.markdown("#### Logging")

    st.write(
        "Validation and blind-prediction requests are logged with a "
        "timestamp, photo identifier, entered order where applicable, "
        "model output, result, and measured inference/comparison time. "
        "Logs on free cloud hosting may not persist across restarts."
    )
