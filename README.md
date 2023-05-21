# Automated Person Removal

## Problem Statement
กลุ่มของเราต้องการทำ Automated Person Removal โดยรับอินพุตเป็นรูปทิวทัศน์ที่มีคนอื่นอยู่ จากนั้นให้ผู้ใช้งานทำการเลือกบริเวณที่จะลบคนออก ซึ่งระบบของเราจะลบคนในบริเวณนั้นให้และเติมสีพื้นหลังที่ลบออกให้ใกล้เคียงกับความเป็นจริงมากที่สุด และให้รูปทิวทัศน์ที่สมบูรณ์คืนมาเป็นเอาท์พุต เนื่องจากการถ่ายในบางครั้งอาจเป็นเพียงโอกาสเดียวที่จะได้บันทึกความทรงจำเหล่านั้นไว้ หากมีคนอื่นอยู่ในรูปอาจทำให้รูปเหล่านั้นมีจุดบกพร่อง นอกจากนี้บุคคลที่ติดมากับรูปของเราอาจไม่ให้ความยินยอมที่จะถูกถ่าย  

## Technical Challenges
- แนวรูปทิวทัศน์ที่หลากหลาย ซึ่งแต่ละแนวรูปก็จะมีความเฉพาะไม่เหมือนกัน ทำให้การเติมพื้นหลังนั้นเป็นไปได้ยาก จะต้องเตรียมรูปให้ครอบคลุมกับขอบเขตของทิวทัศน์ที่จะเติม
- ถ้าหากคนในบริเวณที่เลือกมีความกลมกลืนกับพื้นหลัง จะต้องเลือกขอบเขตของคนได้ถูกต้องและครบถ้วน 
- ถ้าหากมีเงาของคนที่ต้องการจะลบในบริเวณนอกเหนือจากที่เลือกไว้ จะต้องสามารถลบได้ 

## Related Works
- Instance Segmentation: ใช้ Mask R-CNN pretrain on coco dataset 
  - Mask R-CNN https://github.com/matterport/Mask_RCNN
- Image Inpainting
  - deepfillv2 https://github.com/NATCHANONPAN/deepfillv2
  - lama https://github.com/Sanster/lama-cleaner
  - Partial Convolution https://github.com/pakornyodkab/partialconv

## Method and Results
Method
1. instance segmentation หา mask ของคนทั้งหมด
2. ใช้ ui รับ dot ของคนที่ user ต้องการเอาออก
3. หา mask ของคนที่ถูก dot และหาเงาของคนคนนั้นไปใส่ใน mask ผลลัพธ์ที่จะนำไปใช้ตอน inpaint
3. นำ mask และ รูปต้นฉบับ ส่งไปให้ inpainting model ทำการลบคนแล้วเติมพื้นหลัง
5. inpainting model ไปเติมรูปภาพให้สมบูรณ์ แล้วคืนรูปที่ลบคนออกไปแล้วกลับมาให้ ui แสดงผล

Result
- ทำการ evaluate ด้วย L1 loss ของ pytorch
  - DeepfillV2 Pretrained loss 0.8166848302622002
  - DeepfillV2 Fine tune loss 0.8542238303130104
  - Lama Loss 0.19175073754598923
  - Partial Convolution Loss 0.8016573508714017
## Discussion and Future Work
- ถ้า mask มีขนาดเล็ก ไม่ครอบคลุมทั้งหมดของคนจะทำให้ inpainting ไม่ดีเท่าที่ควร
- ถ้าคนอยู่ติดกันหลายๆคน จะมีปัญหาเรื่อง segmentation เงา
- ถ้าคนถือ object อื่นอยู่ จะทำการลบแค่คน ไม่ลบ object ที่ถือออกไปด้วย

## Members
- Pakorn Kongrit 6230409421
- Natchanon Panthuwadeethorn 6231321021
- napat ittidej 6231319821